import csdl_alpha as csdl
import CADDEE_alpha as cd
from CADDEE_alpha import functions as fs
import numpy as np
from functions import map_parametric_wing_pressure, map_right_wing
from pathlib import Path

class PlottingUtil():
    def __init__(self):
        # Initialize CADDEE and import geometry
        rec = csdl.get_current_recorder()
        rec.inline = True

        caddee = cd.CADDEE()
        c172_geom = cd.import_geometry('c172.stp', file_path=Path('./'))
        self.caddee = caddee
        geo_space, displacement_space, pressure_space = define_base_config(caddee, c172_geom)
        self.geo_space = geo_space
        self.displacement_space = displacement_space
        self.pressure_space = pressure_space

    @staticmethod
    def unstack_coefficients(function_set_space:fs.FunctionSetSpace, coefficients):
        coefficients_list = []
        ind = 0
        
        for i, space in function_set_space.spaces.items():
            shape = int(np.prod(space.coefficients_shape))
            coefficients_list.append(coefficients[ind:ind+shape])
            ind += shape
        return coefficients_list
    
    def _make_function(self, function_space, coefficients):
        coefficients = self.unstack_coefficients(function_space, coefficients)
        functions = {}
        for i, function_space in function_space.spaces.items():
            functions[i] = fs.Function(function_space, coefficients[i])
        function_set = fs.FunctionSet(functions)
        return function_set

    def make_geo_function(self, geo_coeffs):
        return self._make_function(self.geo_space, geo_coeffs)
    
    def make_disp_function(self, disp_coeffs):
        return self._make_function(self.displacement_space, disp_coeffs)
    
    def make_pressure_function(self, pressure_coeffs):
        return self._make_function(self.pressure_space, pressure_coeffs)

    def plot_function(self, geo_function:fs.FunctionSet, color_function:fs.FunctionSet=None):
        geo_function.plot_but_good(color=color_function, show=True)










def define_base_config(caddee : cd.CADDEE, c172_geom):
    # Quantities
    skin_thickness = 0.007
    spar_thickness = 0.001
    rib_thickness = 0.001
    num_ribs = 10

    aircraft = cd.aircraft.components.Aircraft(geometry=c172_geom)
    base_config = cd.Configuration(system=aircraft)
    caddee.base_configuration = base_config

    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselag"])
    fuselage = cd.aircraft.components.Fuselage(length=7.49198, geometry=fuselage_geometry)

    # Setup wing component
    wing_geometry = aircraft.create_subgeometry(
        search_names=['MainWing'],
        # The wing coming out of openVSP has some extra surfaces that we don't need
        ignore_names=['0, 8', '0, 9', '0, 14', '0, 15', '1, 16', '1, 17', '1, 22', '1, 23']
    )

    aspect_ratio = csdl.Variable(name="wing_aspect_ratio", value=7.72)
    wing_root_twist = csdl.Variable(name="wing_root_twist", value=np.deg2rad(0))
    wing_tip_twist = csdl.Variable(name="wing_tip_twist", value=np.deg2rad(0))

    # Set design variables for wing
    aspect_ratio.set_as_design_variable(upper=1.5 * 7.72, lower=0.5 * 7.72, scaler=1/8)
    wing_root_twist.set_as_design_variable(upper=np.deg2rad(5), lower=np.deg2rad(-5), scaler=4)
    wing_tip_twist.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=2)

    wing = cd.aircraft.components.Wing( 
        AR=aspect_ratio, S_ref=16.17, 
        taper_ratio=0.73, root_twist_delta=wing_root_twist,
        tip_twist_delta=wing_tip_twist, 
        geometry=wing_geometry
    )
    aircraft.comps["wing"] = wing
    
    # Generate internal geometry
    wing.construct_ribs_and_spars(
        c172_geom,
        num_ribs=num_ribs,
        LE_TE_interpolation="ellipse",
        full_length_ribs=True,
        spanwise_multiplicity=10,
        offset=np.array([0.,0.,.15]),
        finite_te=True,
    )

    # extract relevant geometries
    right_wing = wing.create_subgeometry(search_names=[''], ignore_names=[', 1, ', '_r_', '-'])
    right_wing_oml = wing.create_subgeometry(search_names=['MainWing, 0'])
    left_wing_oml = wing.create_subgeometry(search_names=['MainWing, 1'])
    right_wing_spars = wing.create_subgeometry(search_names=['spar'], ignore_names=['_r_', '-'])
    right_wing_ribs = wing.create_subgeometry(search_names=['rib'], ignore_names=['_r_', '-'])
    wing_oml = wing.create_subgeometry(search_names=['MainWing'])
    wing.quantities.right_wing_oml = right_wing_oml
    wing.quantities.oml = wing_oml



    aero_inds = [11, 12, 19, 20]
    right_inds = [11, 12]
    # wing_aero_functions = {key: wing_oml.functions[key] for key in [11, 12, 19, 20]}


    mono_wing_fs = fs.BSplineSpace(2, (1, 3), (3, int(319*2/16)))
    wing_aero_geo:fs.FunctionSet = c172_geom.declare_component(function_indices=aero_inds)
    # wing_function_space = wing_aero_geo.functions[11].space
    # print(wing_function_space.degree)
    # print(wing_function_space.coefficients_shape)
    # wing_aero_geo.plot()
    # exit()



    # material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=E, G=G, 
                                                density=density, nu=nu)

    # Define thickness functions
    # The ribs and spars have a constant thickness, while the skin has a variable thickness that we will optimize
    thickness_fs = fs.ConstantSpace(2)
    skin_fs = fs.BSplineSpace(2, (2,1), (5,2))
    r_skin_fss = right_wing_oml.create_parallel_space(skin_fs)
    skin_t_coeffs, skin_fn = r_skin_fss.initialize_function(1, value=skin_thickness)
    spar_fn = fs.Function(thickness_fs, spar_thickness)
    rib_fn = fs.Function(thickness_fs, rib_thickness)

    # correlate the left and right wing skin thickness functions - want symmetry
    oml_lr_map = {rind:lind for rind, lind in zip(right_wing_oml.functions, left_wing_oml.functions)}
    wing.quantities.oml_lr_map = oml_lr_map

    # build function set out of the thickness functions
    functions = skin_fn.functions.copy()
    for ind in wing.geometry.functions:
        name = wing.geometry.function_names[ind]
        if "spar" in name:
            functions[ind] = spar_fn
        elif "rib" in name:
            functions[ind] = rib_fn

    for rind, lind in oml_lr_map.items():
        # the v coord is flipped left to right
        functions[lind] = fs.Function(skin_fs, functions[rind].coefficients[:,::-1,:])

    thickness_function_set = fs.FunctionSet(functions)
    wing.quantities.material_properties.set_material(aluminum, thickness_function_set)

    # set skin thickness as a design variable
    skin_t_coeffs.set_as_design_variable(upper=0.05, lower=0.0001, scaler=5e2)
    skin_t_coeffs.add_name('skin_thickness')

    # Spaces for states
    # pressure
    # pressure_function_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, grid_size=(240, 40), conserve=False)
    pressure_function_space = fs.RBFFunctionSpace(num_parametric_dimensions=2, grid_size=(20, 40), radial_function='gaussian', epsilon=10)
    # pressure_mapping = fs.ParametricMapping(map_parametric_wing_pressure, aero_inds)
    pressure_mapping = fs.ParametricMapping(map_right_wing, right_inds)



    # only inds 11 and 12 are needed for the pressure function
    struct_oml_inds = [11, 12]
    struct_oml_fns = {11:wing_oml.functions[11], 12:wing_oml.functions[12]}
    struct_oml = fs.FunctionSet(struct_oml_fns)

    indexed_pressue_function_space = struct_oml.create_parallel_space(pressure_function_space)
    indexed_pressue_function_space.add_parametric_map(pressure_mapping, pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space


    pressure_coeffs, pressure_function = indexed_pressue_function_space.initialize_function(1, value=1)
    pressure_coeffs.add_name('pressure_coefficients')
    wing.quantities.pressure_function = pressure_function



    # displacement
    displacement_space = fs.BSplineSpace(2, (1,1), (3,3))
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)
    wing.quantities.oml_displacement_space = wing_oml.create_parallel_space(
                                                    displacement_space)
    

    displacement_function_space = fs.RBFFunctionSpace(num_parametric_dimensions=2, grid_size=(20, 20), radial_function='gaussian', epsilon=10)
    displacement_mapping = pressure_mapping
    # indexed_displacement_function_space = wing_oml.create_parallel_space(displacement_function_space)
    indexed_displacement_function_space = struct_oml.create_parallel_space(displacement_function_space)
    indexed_displacement_function_space.add_parametric_map(displacement_mapping, displacement_function_space)
    wing.quantities.oml_displacement_space_2 = indexed_displacement_function_space



    # Connect wing to fuselage at the quarter chord
    base_config.connect_component_geometries(fuselage, wing, 0.75 * wing.LE_center + 0.25 * wing.TE_center)

    # base_config.setup_geometry()

    parametric_grid = wing_aero_geo.generate_parametric_grid((5, 1000))
    grid = wing_aero_geo.evaluate(parametric_grid)
    mono_wing_fss = wing_aero_geo.create_parallel_space(mono_wing_fs)
    parametric_map = fs.ParametricMapping(map_parametric_wing_pressure, aero_inds)
    mono_wing_fss.add_parametric_map(parametric_map, mono_wing_fs)
    mapped_parametric_grid = mono_wing_fss._apply_parametric_maps(parametric_grid)
    mono_wing_function = mono_wing_fss.fit_function_set(grid, mapped_parametric_grid)
    
    mono_grid = mono_wing_function.evaluate(parametric_grid)

    fitting_error = np.linalg.norm((grid - mono_grid).value)/np.linalg.norm(grid.value)
    print(f"Fitting error: {fitting_error}")
    
    
    wing.quantities.mono_wing_function = mono_wing_function

    wing_oml_coefficients = wing_oml.stack_coefficients()
    wing_oml_coefficients.add_name('wing_oml_coefficients')

    mono_wing_oml_coefficients = mono_wing_function.stack_coefficients()
    mono_wing_oml_coefficients.add_name('mono_wing_oml_coefficients')

    return mono_wing_fss, indexed_displacement_function_space, indexed_pressue_function_space

