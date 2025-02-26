import csdl_alpha as csdl
import CADDEE_alpha as cd
from CADDEE_alpha import functions as fs
import numpy as np
import os
import sys
import csdml as ml

from functions import map_parametric_wing_pressure, map_right_wing

from femo_alpha.rm_shell.rm_shell_model import RMShellModel
from femo_alpha.fea.utils_dolfinx import readFEAMesh, reconstructFEAMesh
from pathlib import Path

# Settings
couple = False
optimize = True
inline = False
shell = False


# Quantities
skin_thickness = 0.007
spar_thickness = 0.001
rib_thickness = 0.001
mesh_fname = 'c172_tri.msh'
mass = 1000
stress_bound = 1e8
num_ribs = 10
load_factor = 3

# Start recording
rec = csdl.Recorder(inline=True, debug=True)
rec.start()

generator = ml.Generator(recorder=rec)

# Initialize CADDEE and import geometry
caddee = cd.CADDEE()
c172_geom = cd.import_geometry('c172.stp', file_path=Path('./'))

def define_base_config(caddee : cd.CADDEE):
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

    generator.add_input(aspect_ratio)
    generator.add_input(wing_root_twist)
    generator.add_input(wing_tip_twist)

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

    # create pressure function with random coefficients
    # pressure_coeffs_val = np.random.rand(indexed_pressue_function_space.coefficients_shape)
    # pressure_coeffs = csdl.Variable(value=pressure_coeffs, name='pressure_coefficients')
    
    pressure_coeffs, pressure_function = indexed_pressue_function_space.initialize_function(1, value=1)
    pressure_coeffs.add_name('pressure_coefficients')
    generator.add_input(pressure_coeffs, upper=np.ones(pressure_coeffs.shape), lower=-np.ones(pressure_coeffs.shape))
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

    # meshing
    mesh_container = base_config.mesh_container

    # vlm mesh
    vlm_mesh = cd.mesh.VLMMesh()
    wing_chord_surface = cd.mesh.make_vlm_surface(
        wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
        spacing_spanwise="cosine", ignore_camber=True, plot=False,
    )
    wing_chord_surface.project_airfoil_points(oml_geometry=wing_oml)
    vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
    mesh_container["vlm_mesh"] = vlm_mesh

    # beam mesh
    beam_mesh = cd.mesh.BeamMesh()
    beam_disc = cd.mesh.make_1d_box_beam(wing, num_ribs*2-1, 0.5, project_spars=True, spar_search_names=['1', '2'], make_half_beam=True)
    beam_mesh.discretizations["wing"] = beam_disc
    mesh_container["beam_mesh"] = beam_mesh

    # shell meshing
    if True:
        # import shell mesh
        file_path = os.path.dirname(os.path.abspath(__file__))
        wing_shell_mesh = cd.mesh.import_shell_mesh(
            file_path+"/"+mesh_fname, 
            right_wing,
            rescale=[1e-3, 1e-3, 1e-3],
            grid_search_n=5,
            priority_inds=[i for i in right_wing_oml.functions],
            priority_eps=3e-6,
        )
        process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars)

        nodes = wing_shell_mesh.nodal_coordinates
        connectivity = wing_shell_mesh.connectivity
        filename = mesh_fname+"_reconstructed.xdmf"
        if os.path.isfile(filename) and False:
            wing_shell_mesh_fenics = readFEAMesh(filename)
        else:
            # Reconstruct the mesh using the projected nodes and connectivity
            wing_shell_mesh_fenics = reconstructFEAMesh(filename, 
                                                        nodes.value, connectivity)
        # store the xdmf mesh object for shell analysis
        wing_shell_mesh.fea_mesh = wing_shell_mesh_fenics

        wing_shell_mesh_cd = cd.mesh.ShellMesh()
        wing_shell_mesh_cd.discretizations['wing'] = wing_shell_mesh
        mesh_container['shell_mesh'] = wing_shell_mesh_cd


    base_config.setup_geometry()

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

    # wing_plot = wing_oml.plot(show=False, opacity=1, color='red')
    # mono_wing_function.plot(opacity=0.5, additional_plotting_elements=[wing_plot])

    # mono_wing_function.plot_but_good(grid_n=101)
    # exit()

    # for ind in aero_inds:
    #     fun = wing_oml.functions[ind]
    #     print(ind)
    #     fun.evaluate(np.array([0., 0.]), plot=True)
    #     fun.evaluate(np.array([1., 0.]), plot=True)
    #     fun.evaluate(np.array([0., 1.]), plot=True)
    # exit()

    wing_oml_coefficients = wing_oml.stack_coefficients()
    wing_oml_coefficients.add_name('wing_oml_coefficients')
    # generator.add_output(wing_oml_coefficients)

    mono_wing_oml_coefficients = mono_wing_function.stack_coefficients()
    mono_wing_oml_coefficients.add_name('mono_wing_oml_coefficients')
    generator.add_output(mono_wing_oml_coefficients)
    # print(wing_oml_coefficients.shape)
    # print(mono_wing_function.stack_coefficients().shape)
    # exit()
    return pressure_coeffs

def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # Cruise
    pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(2.69268269))
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        range=70 * cd.Units.length.kilometer_to_m,
        mach_number=0.18,
        pitch_angle=pitch_angle, # np.linspace(np.deg2rad(-4), np.deg2rad(10), 15),
    )
    cruise.quantities.pitch_angle = pitch_angle
    cruise.configuration = base_config.copy()
    conditions["cruise"] = cruise

def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions
    wing = caddee.base_configuration.system.comps["wing"]

    # finalize meshes
    cruise:cd.aircraft.conditions.CruiseCondition = conditions["cruise"]
    cruise.finalize_meshes()
    mesh_container = cruise.configuration.mesh_container

    pressure_fn = wing.quantities.pressure_function
    displacement, shell_outputs = run_shell(mesh_container, cruise, pressure_fn, rec=True)

    disp_coeffs = displacement.stack_coefficients()
    disp_coeffs.add_name('displacement_coefficients')
    generator.add_output(disp_coeffs)

def run_shell(mesh_container, condition:cd.aircraft.conditions.CruiseCondition, pressure_function, rec=False):
    wing = condition.configuration.system.comps["wing"]

    # Shell
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    wing_shell_mesh_fenics = wing_shell_mesh.fea_mesh
    element_centers_parametric = wing_shell_mesh.element_centers_parametric
    oml_node_inds = wing_shell_mesh.oml_node_inds
    oml_nodes_parametric = wing_shell_mesh.oml_nodes_parametric

    new_nodes = wing.geometry.evaluate(nodes_parametric)
    new_nodes_oml = new_nodes[oml_node_inds]
    new_nodes_oml.add_name('nodal_geometry')
    generator.add_output(new_nodes_oml)
    node_disp = new_nodes - nodes.reshape((-1,3))

    # transfer aero peressures
    # unique_keys = set()
    # for key, point in oml_nodes_parametric:
    #     unique_keys.add(key)
    # print(unique_keys)
    # for key, function in wing.quantities.oml.functions.items():
    #     print(key)
    # exit()
    pressure_magnitudes = pressure_function.evaluate(oml_nodes_parametric)
    pressure_magnitudes.add_name('nodal_pressure_magnitudes')
    generator.add_output(pressure_magnitudes)
    pressure_normals = wing.quantities.oml.evaluate_normals(oml_nodes_parametric)
    oml_pressures = pressure_normals*csdl.expand(pressure_magnitudes, pressure_normals.shape, 'i->ij')

    shell_pressures = csdl.Variable(value=np.zeros(nodes.shape[1:]))
    shell_pressures = shell_pressures.set(csdl.slice[oml_node_inds], oml_pressures)
    f_input = shell_pressures

    material = wing.quantities.material_properties.material
    element_thicknesses = wing.quantities.material_properties.evaluate_thickness(element_centers_parametric)

    E0, nu0, G0 = material.from_compliance()
    density0 = material.density

    # create node-wise material properties
    nel = connectivity.shape[0]
    E = E0*np.ones(nel)
    E.add_name('E')
    nu = nu0*np.ones(nel)
    nu.add_name('nu')
    density = density0*np.ones(nel)
    density.add_name('density')

    # define boundary conditions
    def clamped_boundary(x):
        eps = 1e-3
        return np.less_equal(x[1], eps)

    # run solver
    shell_model = RMShellModel(mesh=wing_shell_mesh_fenics,
                               shell_bc_func=clamped_boundary,
                               element_wise_material=True,
                               record=rec, # record=true doesn't work with 2 shell instances
                               rho=4)
    shell_outputs = shell_model.evaluate(f_input, 
                                         element_thicknesses, E, nu, density,
                                         node_disp,
                                         debug_mode=False)
    disp_extracted = shell_outputs.disp_extracted

    # fit displacement function
    oml_displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space_2
    oml_displacement_vals = disp_extracted[oml_node_inds]
    oml_displacement_vals.add_name('nodal_displacement')
    generator.add_output(oml_displacement_vals)
    oml_displacement_function = oml_displacement_space.fit_function_set(oml_displacement_vals, oml_nodes_parametric)

    return oml_displacement_function, shell_outputs

def process_elements(wing_shell_mesh, right_wing_oml, right_wing_ribs, right_wing_spars):
    """
    Process the elements of the shell mesh to determine the type of element (rib, spar, skin)
    """

    nodes = wing_shell_mesh.nodal_coordinates
    connectivity = wing_shell_mesh.connectivity

    # figure out type of surface each element is (rib/spar/skin)
    grid_n = 20
    oml_errors = np.linalg.norm(right_wing_oml.evaluate(right_wing_oml.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    rib_errors = np.linalg.norm(right_wing_ribs.evaluate(right_wing_ribs.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)
    spar_errors = np.linalg.norm(right_wing_spars.evaluate(right_wing_spars.project(nodes.value, grid_search_density_parameter=grid_n, plot=False), non_csdl=True) - nodes.value, axis=1)

    element_centers = np.array([np.mean(nodes.value[connectivity[i].astype(int)], axis=0) for i in range(connectivity.shape[0])])

    rib_correction = 1e-4
    element_centers_parametric = []
    oml_inds = []
    rib_inds = []
    spar_inds = []
    for i in range(connectivity.shape[0]):
        inds = connectivity[i].astype(int)
        
        # rib projection is messed up so we use an alternitive approach - if all the points are in an x-z plane, it's a rib
        if np.all(np.isclose(nodes.value[inds, 1], nodes.value[inds[0], 1], atol=rib_correction)):
            rib_inds.append(i)
            continue

        errors = [np.sum(oml_errors[inds]), np.sum(rib_errors[inds]), np.sum(spar_errors[inds])]
        ind = np.argmin(errors)
        if ind == 0:
            oml_inds.append(i)
        elif ind == 1:
            rib_inds.append(i)
        elif ind == 2:
            spar_inds.append(i)
        else:
            raise ValueError('Error in determining element type')

    oml_centers = right_wing_oml.project(element_centers[oml_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    rib_centers = right_wing_ribs.project(element_centers[rib_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    spar_centers = right_wing_spars.project(element_centers[spar_inds], grid_search_density_parameter=5, plot=False, force_reprojection=True)
    oml_inds_copy = oml_inds.copy()

    for i in range(connectivity.shape[0]):
        if oml_inds and oml_inds[0] == i:
            element_centers_parametric.append(oml_centers.pop(0))
            oml_inds.pop(0)
        elif rib_inds and rib_inds[0] == i:
            element_centers_parametric.append(rib_centers.pop(0))
            rib_inds.pop(0)
        elif spar_inds and spar_inds[0] == i:
            element_centers_parametric.append(spar_centers.pop(0))
            spar_inds.pop(0)
        else:
            raise ValueError('Error in sorting element centers')
        
    wing_shell_mesh.element_centers_parametric = element_centers_parametric
    
    oml_node_inds = []
    for c_ind in oml_inds_copy:
        n_inds = connectivity[c_ind].astype(int)
        for n_ind in n_inds:
            if not n_ind in oml_node_inds:
                oml_node_inds.append(n_ind)

    oml_nodes_parametric = right_wing_oml.project(nodes.value[oml_node_inds], grid_search_density_parameter=5)
    wing_shell_mesh.oml_node_inds = oml_node_inds
    wing_shell_mesh.oml_nodes_parametric = oml_nodes_parametric
    wing_shell_mesh.oml_el_inds = oml_inds_copy

def mirror_function(displacement, oml_lr_map):
    for rind, lind in oml_lr_map.items():
        displacement.functions[lind].coefficients = displacement.functions[rind].coefficients[:,::-1,:]

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_dv_values(fname, group):
    inputs = csdl.inline_import(fname, group)
    recorder = csdl.get_current_recorder()
    dvs = recorder.design_variables
    for var in dvs:
        var_name = var.name
        var.set_value(inputs[var_name].value)
        scale = 1/np.linalg.norm(inputs[var_name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])

pressure_coeffs = define_base_config(caddee)
rec.inline = inline
define_conditions(caddee)
define_analysis(caddee)
csdl.save_optimization_variables()


generator.generate(filename='struct_opt_geo_test_05', n_samples=2)

exit()

