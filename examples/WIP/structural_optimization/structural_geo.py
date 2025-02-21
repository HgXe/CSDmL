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
    generator.add_output(wing_oml_coefficients)

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

    # # run VLM 
    # vlm_outputs, implicit_disp_coeffs = run_vlm([mesh_container], [cruise])
    # forces = vlm_outputs.surface_force[0][0]
    # Cp = vlm_outputs.surface_spanwise_Cp[0][0]
    
    # # trim the aircraft
    # pitch_angle = cruise.quantities.pitch_angle
    # z_force = -forces[2]*csdl.cos(pitch_angle) + forces[0]*csdl.sin(pitch_angle)
    # residual = z_force - mass*9.81*load_factor
    # trim_solver = csdl.nonlinear_solvers.BracketedSearch()
    # trim_solver.add_state(pitch_angle, residual, (np.deg2rad(0), np.deg2rad(10)))
    # with HiddenPrints():
    #     # The vlm solver prints stuff every time it's called and it annoys me
    #     trim_solver.run()

    # # fit pressure function to trimmed VLM results
    # # we can actually do this before the trim if we wanted, it would be updated automatically
    # pressure_fn = fit_pressure_fn(mesh_container, cruise, Cp)

    # return

    pressure_fn = wing.quantities.pressure_function
    displacement, shell_outputs = run_shell(mesh_container, cruise, pressure_fn, rec=True)

    disp_coeffs = displacement.stack_coefficients()
    disp_coeffs.add_name('displacement_coefficients')
    generator.add_output(disp_coeffs)

    # # Run structural analysis
    # if shell:
    #     displacement, shell_outputs = run_shell(mesh_container, cruise, pressure_fn, rec=True)
    #     max_stress:csdl.Variable = shell_outputs.aggregated_stress
    #     wing_mass:csdl.Variable = shell_outputs.mass
    # else:
    #     displacement, max_stress, wing_mass = run_beam(mesh_container, cruise, pressure_fn)

    # mirror_function(displacement, wing.quantities.oml_lr_map)

    # max_stress.set_as_constraint(upper=stress_bound, scaler=1e-8)
    # max_stress.add_name('max_stress')
    # wing_mass.set_as_objective(scaler=1e-2)
    # wing_mass.add_name('wing_mass')
    # wing.quantities.oml_displacement = displacement
    # wing.quantities.pressure_function = pressure_fn

    # # Solver for aerostructural coupling
    # # we iterate between the VLM and structural analysis until the displacement converges
    # if couple:
    #     coeffs = displacement.stack_coefficients()
    #     disp_res = implicit_disp_coeffs[0] - coeffs
    #     solver = csdl.nonlinear_solvers.Jacobi(max_iter=10, tolerance=1e-6)
    #     solver.add_state(implicit_disp_coeffs[0], disp_res)
    #     solver.run()

def run_vlm(mesh_containers, conditions):
    # implicit displacement input
    wing = conditions[0].configuration.system.comps["wing"]
    displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    implicit_disp_coeffs = []
    implicit_disp_fns = []
    for i in range(len(mesh_containers)):
        coeffs, function = displacement_space.initialize_function(3, implicit=True)
        implicit_disp_coeffs.append(coeffs)
        implicit_disp_fns.append(function)

    # set up VLM analysis
    nodal_coords = []
    nodal_vels = []

    for mesh_container, condition, disp_fn in zip(mesh_containers, conditions, implicit_disp_fns):
        transfer_mesh_para = disp_fn.generate_parametric_grid((5, 5))
        transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)
        transfer_mesh_disp = disp_fn.evaluate(transfer_mesh_para)
        
        wing_lattice = mesh_container["vlm_mesh"].discretizations["wing_chord_surface"]
        wing_lattic_coords = wing_lattice.nodal_coordinates

        map = fs.NodalMap()
        weights = map.evaluate(csdl.reshape(wing_lattic_coords, (np.prod(wing_lattic_coords.shape[0:-1]), 3)), transfer_mesh_phys)
        wing_camber_mesh_displacement = (weights @ transfer_mesh_disp).reshape(wing_lattic_coords.shape)
        
        nodal_coords.append(wing_lattic_coords + wing_camber_mesh_displacement)
        nodal_vels.append(wing_lattice.nodal_velocities) # the velocities should be the same for every node in this case

    if len(nodal_coords) == 1:
        nodal_coordinates = nodal_coords[0]
        nodal_velocities = nodal_vels[0]
    else:
        nodal_coordinates = csdl.vstack(nodal_coords)
        nodal_velocities = csdl.vstack(nodal_vels)

    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
        num_interp=120,
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])

    vlm_outputs = vlm_solver(
        mesh_list=[nodal_coordinates],
        mesh_velocity_list=[nodal_velocities],
        atmos_states=conditions[0].quantities.atmos_states,
        airfoil_alpha_stall_models=[alpha_stall_model],
        airfoil_Cd_models=[Cd_model],
        airfoil_Cl_models=[Cl_model],
        airfoil_Cp_models=[Cp_model], 
    )

    return vlm_outputs, implicit_disp_coeffs

def fit_pressure_fn(mesh_container, condition, spanwise_Cp):
    wing:cd.Component = condition.configuration.system.comps["wing"]
    vlm_mesh = mesh_container["vlm_mesh"]
    rho = condition.quantities.atmos_states.density
    v_inf = condition.parameters.speed
    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para

    spanwise_p = spanwise_Cp * 0.5 * rho * v_inf**2
    spanwise_p = csdl.blockmat([[spanwise_p[:, 0:120].T()], [spanwise_p[:, 120:].T()]])

    pressure_indexed_space : fs.FunctionSetSpace = wing.quantities.pressure_space
    pressure_function = pressure_indexed_space.fit_function_set(
        values=spanwise_p.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
        regularization_parameter=1e-4,
    )

    # fitting_error = np.linalg.norm(((spanwise_p.reshape((-1, )) - pressure_function.evaluate(airfoil_upper_nodes+airfoil_lower_nodes))/spanwise_p.reshape((-1,))).value)
    # print(f"Pressure function fitting error: {fitting_error}")

    # oml:fs.FunctionSet = wing.quantities.mono_wing_function
    # oml.plot_but_good(color=pressure_function, grid_n=101)
    # exit()

    pressure_function_coefficients = pressure_function.stack_coefficients()
    pressure_function_coefficients.add_name('pressure_function')
    generator.add_output(pressure_function_coefficients)

    # pressure_eval_pts = airfoil_upper_nodes + airfoil_lower_nodes
    # for i in range(len(pressure_eval_pts)):
    #     pressure_eval_pts[i] = [pressure_eval_pts[i][0], pressure_eval_pts[i][1][0, 0], pressure_eval_pts[i][1][0, 1]]
    # pressure_eval_pts = np.array(pressure_eval_pts)
    # np.savetxt('pressure_eval_pts.txt', pressure_eval_pts)
    # exit()
    para_grid = np.array(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 25))).reshape((2, -1)).T
    # np.savetxt('pressure_eval_pts_2.txt', para_grid)
    # exit()
    para_grid = [(-1, para_grid[i].reshape(1,-1)) for i in range(para_grid.shape[0])]
    # pressure_eval = pressure_function.evaluate(airfoil_upper_nodes+airfoil_lower_nodes)
    pressure_eval = pressure_function.evaluate(para_grid)
    # wing.quantities.mono_wing_function.evaluate(para_grid, plot=True)
    # exit()
    pressure_eval.add_name('pressure_eval')
    generator.add_output(pressure_eval)

    # geo_eval = wing.geometry.evaluate(airfoil_upper_nodes+airfoil_lower_nodes)
    geo_eval = wing.quantities.mono_wing_function.evaluate(para_grid)
    geo_eval.add_name('geo_eval')
    generator.add_output(geo_eval)

    return pressure_function

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
    node_disp = wing.geometry.evaluate(nodes_parametric) - nodes.reshape((-1,3))

    # transfer aero peressures
    # unique_keys = set()
    # for key, point in oml_nodes_parametric:
    #     unique_keys.add(key)
    # print(unique_keys)
    # for key, function in wing.quantities.oml.functions.items():
    #     print(key)
    # exit()
    pressure_magnitudes = pressure_function.evaluate(oml_nodes_parametric)    
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
    oml_displacement_function = oml_displacement_space.fit_function_set(disp_extracted[oml_node_inds], oml_nodes_parametric)

    return oml_displacement_function, shell_outputs

def run_beam(mesh_container, condition, pressure_fn):
    wing = condition.configuration.system.comps["wing"]
    beam_mesh = mesh_container["beam_mesh"]
    wing_box = beam_mesh.discretizations["wing"]
    aluminum = wing.quantities.material_properties.material

    box_beam = mesh_container["beam_mesh"].discretizations["wing"]
    beam_nodes = box_beam.nodal_coordinates

    box_cs = af.CSBox(
        ttop=wing_box.top_skin_thickness,
        tbot=wing_box.bottom_skin_thickness,
        tweb=wing_box.shear_web_thickness,
        height=wing_box.beam_height,
        width=wing_box.beam_width,
    )
    beam = af.Beam(
        name="wing_beam", 
        mesh=wing_box.nodal_coordinates, 
        cs=box_cs,
        material=aluminum,
    )

    # transfer aero forces
    right_wing_oml_inds = list(wing.quantities.right_wing_oml.functions)
    force_magnitudes, force_para_coords = pressure_fn.integrate(wing.geometry, grid_n=30, indices=right_wing_oml_inds)
    force_coords = wing.geometry.evaluate(force_para_coords)
    force_normals = wing.geometry.evaluate_normals(force_para_coords)
    force_vectors = force_normals*csdl.expand(force_magnitudes.flatten(), force_normals.shape, 'i->ij')

    mapper = fs.NodalMap(weight_eps=5)
    force_map = mapper.evaluate(force_coords, beam_nodes.reshape((-1, 3)))
    beam_forces = force_map.T() @ force_vectors

    beam_forces_plus_moments = csdl.Variable(shape=(beam_forces.shape[0], 6), value=0)
    beam_forces_plus_moments = beam_forces_plus_moments.set(
        csdl.slice[:, 0:3], beam_forces
    )

    beam.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
    beam.add_load(beam_forces_plus_moments)

    frame = af.Frame()
    frame.add_beam(beam)

    struct_solution = frame.evaluate()

    beam_displacement = struct_solution.get_displacement(beam)
    beam_stress = struct_solution.get_stress(beam)

    # transfer displacements to oml
    mapper = fs.NodalMap(weight_eps=5, weight_to_be_normalized=True)
    oml_mesh_parametric = wing.quantities.right_wing_oml.generate_parametric_grid((10, 10))
    oml_mesh_phys = wing.geometry.evaluate(oml_mesh_parametric)
    disp_map = mapper.evaluate(oml_mesh_phys, beam_nodes.reshape((-1, 3)))
    disp = disp_map @ beam_displacement
    oml_displacement_space:fs.FunctionSetSpace = wing.quantities.oml_displacement_space
    oml_displacement_function = oml_displacement_space.fit_function_set(disp, oml_mesh_parametric)

    # get wing mass
    mass_fn = af.FrameMass()
    mass_fn.add_beam(beam)
    wing_mass_prop = mass_fn.evaluate()
    wing_mass = wing_mass_prop.mass

    return oml_displacement_function, beam_stress, wing_mass

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


# for sampling, I want n different samples of the pressure coefficients, where n is the number of pressure coefficients
# in this case, n = 20*40 = 800
# I want to distribute these coefficient vectors across the samples of the design variables - root twist, tip twist, aspect ratio
# If we do 20 samples of each, we will have 20*20*20 = 8000 samples, so 10 samples per coefficient vector
samples_per_dim = 20
filename = 'struct_opt_geo_test_03'
# function for running the model
function = generator._build_generator_function()

# make input dict sans coefficients
inputs = {input:bounds for input, bounds in generator.inputs.items() if input != pressure_coeffs}

# Generate the 800 coefficient vectors - want these to be linearly independent - values between -1 and 1
# a random matrix is usually full rank
dim = np.prod(pressure_coeffs.shape)
coefficients = np.random.rand(dim, dim)*2 - 1

# Generate the 8000 samples - latin hypercube sampling
from scipy.stats import qmc
dims = []
for input, bounds in inputs.items():
    print(input.name)
    dims.append(np.prod(input.shape))
    # apply default bounds
    if bounds[0] is None:
        bounds[0] = np.ones(input.shape) * 1
    if bounds[1] is None:
        bounds[1] = np.ones(input.shape) * 0

upper = np.hstack([inputs[input][0].flatten() for input in inputs]).flatten()
lower = np.hstack([inputs[input][1].flatten() for input in inputs]).flatten()
n_samples = samples_per_dim ** sum(dims)
samples = qmc.LatinHypercube(d=sum(dims)).random(n_samples)
scaler = upper - lower
offset = lower
samples = samples * scaler + offset

# samples should be a 2D array with n_samples rows and sum(dims) columns - in this case, 8000 rows and 3 columns
# now we add the pressure coefficient vectors to the samples, final shape should be 8000 rows and 803 columns
# the coefficients should be added to the end of the samples, and repeated 10 times
# it's on the end because it was added last.
samples = np.hstack([samples, np.tile(coefficients, (n_samples//dim, 1))])

# re-make dims array to include the coefficients
dims.append(dim)

print_interval = n_samples // 10

import time

for n, sample in enumerate(samples):
    if n % print_interval == 0:
        print(f'Generating samples {n}-{min(n+print_interval, n_samples)} of {n_samples}')

    ind = 0
    in_dict = {}
    for i, input in enumerate(generator.inputs):
        in_dict[input] = sample[ind:ind+dims[i]].reshape(input.shape)
        ind += dims[i]

    start = time.time()
    result = function(in_dict)
    end = time.time()
    print(f'Function call took {end-start} seconds')
    generator._export_h5py(filename, {**in_dict, **result}, f'sample_{n}')

    # just going to run the first 10 to make sure everything is working
    if n == 9:
        break

exit()

# generator.generate(filename='struct_opt_geo_test_01', samples_per_dim=1)
# exit()

# fname = 'structural_opt_beam_test'
# if optimize:
#     from modopt import CSDLAlphaProblem
#     from modopt import PySLSQP

    
#     # If you have a GPU, you can set gpu=True - but it may not be faster
#     # I think this is because the ml airfoil model can't be run on the GPU when Jax is using it
#     # sim = csdl.experimental.JaxSimulator(rec, gpu=False, save_on_update=True, filename=fname)
#     sim = csdl.experimental.PySimulator(rec)
#     sim.compute_totals()

#     # If you don't have jax installed, you can use the PySimulator instead (it's slower)
#     # To install jax, see https://jax.readthedocs.io/en/latest/installation.html
#     # sim = csdl.experimental.PySimulator(rec)

#     # It's a good idea to check the totals of the simulator before running the optimizer
#     # sim.check_totals()
#     # exit()

#     prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)
#     optimizer = PySLSQP(prob, solver_options={'maxiter':200, 'acc':1e-6})

#     # Solve your optimization problem
#     optimizer.solve()
#     optimizer.print_results()
#     csdl.inline_export(fname+'_final')


# # Plotting
# # load dv values and perform an inline execution to get the final results
# load_dv_values(fname+'_final.hdf5', 'inline')
# rec.execute()
# wing = caddee.base_configuration.system.comps["wing"]
# mesh = wing.quantities.oml.plot_but_good(color=wing.quantities.material_properties.thickness)
# wing.quantities.oml.plot_but_good(color=wing.quantities.oml_displacement)
# wing.quantities.oml.plot_but_good(color=wing.quantities.pressure_function)


# csdl.visualize_graph()