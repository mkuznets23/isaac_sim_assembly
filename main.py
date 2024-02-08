from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import delete_prim
# from omni.isaac.core.prims import GeometryPrim
from omni.physx.scripts import utils
# from omni.isaac.core.materials.physics_material import PhysicsMaterial
# from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade
import omni
import time
import numpy as np
from pxr import UsdShade, Sdf, Gf, UsdLux

class FrankaPlaying(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)

        self._num_arms = 4
        self._objs = []
        self._objs_kit1_second = []
        self._objs_kit2_second = []
        self._robots = []
        self._prim_paths = []
        self._z_offset = 0.8
        self._stack_offest = self._z_offset + 0.05
        self._x_offset = 0.1
        self._kit_offset = 0.4
        self._y_offset = 0.45

        self._arm_positions = np.array([[-0.25 , 0 , self._z_offset],
                                        [-0.25 , 0.9 , self._z_offset],
                                        [-1.45 , 0, self._z_offset],
                                        [-1.45 , 0.9 , self._z_offset]])
        
        self._obj_paths = ["/home/misha/Documents/AME547/STLs V4/Tray_1.usd",    #0
                           "/home/misha/Documents/AME547/STLs V4/Tray_2.usd",    #1
                           "/home/misha/Documents/AME547/STLs V4/Chassis.usd",   #2
                           "/home/misha/Documents/AME547/STLs V4/Motor.usd",     #3
                           "/home/misha/Documents/AME547/STLs V4/Motor.usd",     #4
                           "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #5
                           "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #6
                           "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #7
                           "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #8
                           "/home/misha/Documents/AME547/STLs V4/PCB.usd",       #9
                           "/home/misha/Documents/AME547/STLs V4/Battery.usd",   #10
                           "/home/misha/Documents/AME547/STLs V4/Cover.usd",     #11
                           "/home/misha/Documents/AME547/STLs V4/metal_table.usd",
                           "/home/misha/Documents/AME547/STLs V4/metal_table.usd",
                           "/home/misha/Documents/AME547/STLs V4/metal_table.usd",
                           "/home/misha/Documents/AME547/STLs V4/metal_table.usd",
                           "/home/misha/Documents/AME547/STLs V4/wood_table.usd"]

        # self._obj_paths_kits_second = ["/home/misha/Documents/AME547/STLs V4/Tray_1.usd",    #0
        #                    "/home/misha/Documents/AME547/STLs V4/Tray_2.usd",    #1
        #                    "/home/misha/Documents/AME547/STLs V4/Chassis.usd",   #2
        #                    "/home/misha/Documents/AME547/STLs V4/Motor.usd",     #3
        #                    "/home/misha/Documents/AME547/STLs V4/Motor.usd",     #4
        #                    "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #5
        #                    "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #6
        #                    "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #7
        #                    "/home/misha/Documents/AME547/STLs V4/Wheel.usd",     #8
        #                    "/home/misha/Documents/AME547/STLs V4/PCB.usd",       #9
        #                    "/home/misha/Documents/AME547/STLs V4/Battery.usd",   #10
        #                    "/home/misha/Documents/AME547/STLs V4/Cover.usd",]     #11
                                                                                                              # Raised 17
                                                                                                                # Flat 18

        self._material_paths = ["http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Wall_Board/Cardboard.mdl", #0
                                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Wall_Board/Cardboard.mdl", #1
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Plastic/Polypropylene_Cloudy.mdl", #2
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Metal/Metal_Cast.mdl", #3
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Metal/Metal_Cast.mdl", #4
                                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Plastics/Rubber_Smooth.mdl", #5
                                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Plastics/Rubber_Smooth.mdl", #6
                                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Plastics/Rubber_Smooth.mdl", #7
                                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Plastics/Rubber_Smooth.mdl", #8
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Plastic/PCB_Solder_Mask.mdl", #9
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Plastic/PCB_Solder_Mask.mdl", #10
                                "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Plastic/Polypropylene_Cloudy.mdl"]#11
 
        
        self._material_names = ["Cardboard", #0
                                "Cardboard", #1
                                "Polypropylene_Cloudy_Rough", #2
                                "Aluminum_Cast_Shiny", #3
                                "Aluminum_Cast_Shiny", #4
                                "Rubber_Smooth", #5
                                "Rubber_Smooth", #6
                                "Rubber_Smooth", #7
                                "Rubber_Smooth", #8
                                "PCB_Solder_Mask", #9
                                "PCB_Solder_Mask_Blue_Fast", #10
                                "Polypropylene_Cloudy_Rough"] #11
                                
        self._obj_positions = np.array([[self._x_offset  + 0 , self._y_offset + self._kit_offset+ 0 , self._z_offset + 0.01],
                                        [self._x_offset + 0 -1.95 , self._y_offset + self._kit_offset , self._z_offset + 0.01],
                                        [self._x_offset , self._y_offset+ self._kit_offset - 0.012 , self._stack_offest + 0.02],
                                        [self._x_offset + -0.0925 -1.95 , self._y_offset+ self._kit_offset + 0.035 , self._stack_offest + 0.02],
                                        [self._x_offset + 0.0925 -1.95 , self._y_offset+ self._kit_offset + 0.035 , self._stack_offest + 0.02],
                                        [self._x_offset + -0.090 -1.95 , self._y_offset + self._kit_offset+ 0.145 , self._stack_offest + 0.03],
                                        [self._x_offset + 0.090 -1.95 , self._y_offset + self._kit_offset+ 0.145 , self._stack_offest + 0.03],
                                        [self._x_offset + -0.090 -1.95 , self._y_offset + self._kit_offset+ -0.145 , self._stack_offest + 0.03],
                                        [self._x_offset + 0.090 -1.95 , self._y_offset + self._kit_offset+ -0.145 , self._stack_offest + 0.03],
                                        [self._x_offset  + 0 , self._y_offset + self._kit_offset+ -0.1125 , self._stack_offest],
                                        [self._x_offset  + 0 , self._y_offset + self._kit_offset+ -0.170 , self._stack_offest - 0.02],
                                        [self._x_offset  + 0 , self._y_offset + self._kit_offset+ 0.230 , self._stack_offest + 0.02],
                                        [-0.25 , -500 , 0.8],
                                        [-0.25 , -500 , 0.8],
                                        [-1.545 , -500 , 0.8],
                                        [-1.545 , -500 , 0.8],
                                        [-0.9 , 0.45 , 0]])
                                        # [-0.25 , 0 , 0.8],
                                        # [-0.25 , 0.9 , 0.8],
                                        # [-1.545 , 0 , 0.8],
                                        # [-1.545 , 0.9 , 0.8],
                                        # [-0.9 , 0.45 , 0]])
        
        return

    # Here we setup all the assets that we care about in this task.
    
    def set_up_scene(self, scene):

        super().set_up_scene(scene)

        scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        mtl_created_list = []
        for i in range(len(self._obj_paths)):

            add_reference_to_stage(usd_path = self._obj_paths[i] , prim_path = "/World/Obj_" + str(i))
            self._prim_paths.append("/World/Obj_" + str(i))

            if i < 12:

                prim_view = RigidPrim(
                    prim_path = "/World/Obj_" + str(i),
                    name = "/World/Obj_Name_" + str(i),
                    position = self._obj_positions[i,:]
                    )
                if i == 0 or i == 1:
                    prim_view.set_mass(1.000)
                elif i == 2:
                    prim_view.set_mass(0.005)
                else:
                    prim_view.set_mass(0.0045)

                utils.setRigidBody(stage.GetPrimAtPath("/World/Obj_" + str(i)), "convexDecomposition", False)
                self._objs.append(scene.add(prim_view))

                omni.kit.commands.execute(
                    "CreateAndBindMdlMaterialFromLibrary",
                    mdl_name=self._material_paths[i],
                    mtl_name=self._material_names[i],
                    mtl_created_list=mtl_created_list,
                 )

                mtl_prim = stage.GetPrimAtPath(mtl_created_list[len(mtl_created_list)-1])
                mat_shade = UsdShade.Material(mtl_prim)
                prim = stage.GetPrimAtPath("/World/Obj_" + str(i))
                UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)

                if i in [0,2,9,10,11]:
                    add_reference_to_stage(usd_path = self._obj_paths[i] , prim_path = "/World/Obj2_" + str(i))
                    #self._prim_paths.append("/World/Obj2_" + str(i))
                    prim_view = RigidPrim(
                        prim_path = "/World/Obj2_" + str(i),
                        name = "/World/Obj2_Name_" + str(i),
                        position = self._obj_positions[i,:] + np.array([0,1,0]) #adding offset
                        )
                    if i == 0 or i == 1:
                        prim_view.set_mass(1.000)
                    elif i == 2:
                        prim_view.set_mass(0.005)
                    else:
                        prim_view.set_mass(0.0045)

                    utils.setRigidBody(stage.GetPrimAtPath("/World/Obj2_" + str(i)), "convexDecomposition", False)
                    self._objs_kit1_second.append(scene.add(prim_view))

                    mtl_prim = stage.GetPrimAtPath(mtl_created_list[len(mtl_created_list)-1])
                    mat_shade = UsdShade.Material(mtl_prim)
                    prim = stage.GetPrimAtPath("/World/Obj2_" + str(i))
                    UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
                
                if i in [1,3,4,5,6,7,8]:
                    add_reference_to_stage(usd_path = self._obj_paths[i] , prim_path = "/World/Obj2_" + str(i))
                    #self._prim_paths.append("/World/Obj2_" + str(i))
                    prim_view = RigidPrim(
                        prim_path = "/World/Obj2_" + str(i),
                        name = "/World/Obj2_Name_" + str(i),
                        position = self._obj_positions[i,:] + np.array([0,1,0]) #adding offset
                        )
                    if i == 0 or i == 1:
                        prim_view.set_mass(1.000)
                    elif i == 2:
                        prim_view.set_mass(0.005)
                    else:
                        prim_view.set_mass(0.0045)

                    utils.setRigidBody(stage.GetPrimAtPath("/World/Obj2_" + str(i)), "convexDecomposition", False)
                    self._objs_kit2_second.append(scene.add(prim_view))

                    mtl_prim = stage.GetPrimAtPath(mtl_created_list[len(mtl_created_list)-1])
                    mat_shade = UsdShade.Material(mtl_prim)
                    prim = stage.GetPrimAtPath("/World/Obj2_" + str(i))
                    UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)

            elif i > 11 and i < 16:
                #continue #skip placing thor tables
                if i ==14 or i == 15:
                    prim_view = RigidPrim(
                    prim_path = "/World/Obj_" + str(i),
                    name = "/World/Obj_Name_" + str(i),
                    position = self._obj_positions[i,:],
                    scale = np.array([1 , 1.17 , 1]),
                    orientation = np.array([0,0,0,1])
                    )
                else:
                    prim_view = RigidPrim(
                    prim_path = "/World/Obj_" + str(i), 
                    name = "/World/Obj_Name_" + str(i),
                    position = self._obj_positions[i,:],
                    scale = np.array([1 , 1.17 , 1])
                    )

                prim_view.set_mass(1000.000)
                #utils.setRigidBody(stage.GetPrimAtPath("/World/Obj_" + str(i)), "convexDecomposition", False)

                self._objs.append(scene.add(prim_view))

            else:
                prim_view = RigidPrim(
                    prim_path = "/World/Obj_" + str(i),
                    name = "/World/Obj_Name_" + str(i),
                    position = self._obj_positions[i,:],
                    orientation = np.array([1 , 0 , 0 , 1]),
                    scale = np.array([1.26 , 1 , 1])
                    )
                prim_view.set_mass(1000.000)
                utils.setRigidBody(stage.GetPrimAtPath("/World/Obj_" + str(i)), "convexDecomposition", False)
                self._objs.append(scene.add(prim_view))

                mtl_prim = stage.GetPrimAtPath(mtl_created_list[3])
                mat_shade = UsdShade.Material(mtl_prim)
                prim = stage.GetPrimAtPath("/World/Obj_" + str(i))
                UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)

        self._objs.append(scene.add(DynamicCuboid(
                                    prim_path = "/World/Raised",
                                    name = "Raised",
                                    position = np.array([-1.06 , 0.45 , 0.81]),
                                    scale = np.array([0.085 , 0.25 , 0.1])
                                )))
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
        mat_shade = UsdShade.Material(mtl_prim)
        prim = stage.GetPrimAtPath("/World/Raised")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)

        self._objs.append(scene.add(DynamicCuboid(
                                    prim_path = "/World/Platform",
                                    name = "Platform",
                                    position = np.array([-0.68 , 0.45 , 0.81]),
                                    scale = np.array([0.28 , 0.28 , 0.01])
                                )))
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
        mat_shade = UsdShade.Material(mtl_prim)
        prim = stage.GetPrimAtPath("/World/Platform")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
        
        for i in range(self._num_arms):
                self._robots.append(scene.add(Franka(prim_path="/World/Franka_" + str(i),
                                                name="Franka_" + str(i),
                                                position = self._arm_positions[i , :])))

        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor0")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor0"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor0",
            name = "conveyor0",
            position = np.array([-0.14112 , 1.5 , 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor0 = scene.add(prim_view)

        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor1")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor1"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor1",
            name = "conveyor1",
            position = np.array([-0.14112 , 3.52 , 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor1 = scene.add(prim_view)
        
        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor2")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor2"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor2",
            name = "conveyor2",
            position = np.array([-2.62588 , 1.5 , 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor0 = scene.add(prim_view)

        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor3")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor3"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor3",
            name = "conveyor3",
            position = np.array([-2.62588 , 3.52 , 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor1 = scene.add(prim_view)

        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor4")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor4"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor4",
            name = "conveyor4",
            position = np.array([-0.14112 , -0.50646, 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor0 = scene.add(prim_view)

        add_reference_to_stage(usd_path = "/home/misha/Downloads/conveyor.usd" , prim_path = "/World/conveyor5")
        utils.setRigidBody(stage.GetPrimAtPath("/World/conveyor5"), "convexDecomposition", False)
        prim_view = RigidPrim(
            prim_path = "/World/conveyor5",
            name = "conveyor5",
            position = np.array([-2.62588 , -0.50646, 0]),
            #orientation = np.array([0.70711,0,0,0.70711])
            )
        prim_view.set_mass(1000)
        self._conveyor0 = scene.add(prim_view)

        scene.add(DynamicCuboid(
                                    prim_path = "/World/Col1",
                                    name = "Col1",
                                    position = np.array([-0.32682 , 0 , 0.4]),
                                    scale = np.array([0.3 , 0.3 , 0.8])
                                ))
        
        scene.add(DynamicCuboid(
                                    prim_path = "/World/Col2",
                                    name = "Col2",
                                    position = np.array([-0.32682 , 0.89284 , 0.4]),
                                    scale = np.array([0.3 , 0.3 , 0.8])
                                ))
        
        scene.add(DynamicCuboid(
                                    prim_path = "/World/Col3",
                                    name = "Col3",
                                    position = np.array([-1.4797 , 0.89284 , 0.4]),
                                    scale = np.array([0.3 , 0.3 , 0.8])
                                ))
        
        scene.add(DynamicCuboid(
                                    prim_path = "/World/Col4",
                                    name = "Col4",
                                    position = np.array([-1.4797 , 0 , 0.4]),
                                    scale = np.array([0.3 , 0.3 , 0.8])
                                ))
        
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[3])
        mat_shade = UsdShade.Material(mtl_prim)
        prim = stage.GetPrimAtPath("/World/Col1")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
        prim = stage.GetPrimAtPath("/World/Col2")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
        prim = stage.GetPrimAtPath("/World/Col3")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
        prim = stage.GetPrimAtPath("/World/Col4")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)

        #Set ground plane material
        omni.kit.commands.execute(
                    "CreateAndBindMdlMaterialFromLibrary",
                    # mdl_name="http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
                    # mtl_name="Ceramic_Tiles_Diamond_White_Matte",
                    #  mdl_name="http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Plastic/PCB_Solder_Mask.mdl",
                    #  mtl_name="PCB_Solder_Mask_White",
                    #  mdl_name="http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Paseo.mdl",
                    #  mtl_name="Ceramic_Tiles_Paseo_Gray",
                    #  mdl_name="http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Wood/Wood_Tiles_Ash.mdl",
                    #  mtl_name="Wood_Tiles_Ash_Brickbond",
                     mdl_name="http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/vMaterials_2/Wood/Wood_Tiles_Oak_Mountain.mdl",
                     mtl_name="Wood_Tiles_Oak_Mountain",
                    #  mdl_name="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Wood/Parquet_Floor.mdl",
                    # mtl_name="Parquet_Floor",
                    mtl_created_list=mtl_created_list,
                 )
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[-1])
        mat_shade = UsdShade.Material(mtl_prim)
        
        prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        UsdShade.MaterialBindingAPI(prim).Bind(mat_shade, UsdShade.Tokens.strongerThanDescendants)
        
        #add more light
        l = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/myLight"))
        l.CreateExposureAttr(0.6)
        l.CreateIntensityAttr(50000)
        l.CreateRadiusAttr(1)
        # l.AddTranslateOp()
        XFormPrim("/World/myLight").set_local_pose(translation=np.array([-1,0.45,6.6]))

        XFormPrim("/World/defaultGroundPlane/SphereLight").set_local_pose(translation=np.array([-1,0.45,10]))

        return

    def veloc_control(self , robot_num , desired_position , desired_time , max_desired_velocity):

        acceleration = 0.8
        t_step = 1/200
        curr_time = 0
        veloc_array = self._robots[robot_num].get_joint_velocities()

        while curr_time != desired_time:
            if curr_time <= desired_time / 3:
                curr_v = acceleration * curr_time
            elif curr_time > desired_time / 3 and curr_time < 2 * desired_time / 3:
                curr_v = max_desired_velocity
            else:
                curr_v = max_desired_velocity -  acceleration * (desired_time)

    # Called before each physics step,
    def pre_step(self, control_index, simulation_time):
        return

    # Called after each reset,
    def post_reset(self):
        for i in range(self._num_arms):
            self._robots[i].gripper.set_joint_positions(self._robots[i].gripper.joint_opened_positions)
        return


class HelloWorld(BaseSample):

    def __init__(self) -> None:
        super().__init__()
        self._task = FrankaPlaying(name="Primary_Task")
        # stage = omni.usd.get_context().get_stage()

        self._task_index = np.array([0,0,0,0])
        self._in_use = np.array([0,0,0,0])
        self._in_use_sum = 0
        self._physics_step_counter = 0
        self._wait_time = np.array([0,0,0,0])
        self._gripping = np.array([0,0,0,0])
        self._is_gripping = np.array([-1,-1,-1,-1])
        self._fixed_prims = []
        self._kit1_ready = 0  #flag to tell conveyor to deliver another kit
        self._kit2_ready = 0
        self._need_fix1 = 0 #kit 1 needs to fix its items
        self._need_fix2 = 0#kit 2 needs to fix its items
        self._positions = self._task._obj_positions
        # Task Index -> [type , goal_type , obj_ind , x , y , z , q1 , q2 , q3 , q4 , start_arm? , f_ob1 , f_obj2 , last_task?]
                        # 0         1          2      3   4   5    6   7    8    9       10         11      12          13
        self._task_array = np.zeros((4 , 50 , 14))

        h_a_p = 1.40
        y_a_p = 0.5
        x_a_p = -0.25
        h_a_pl = 0.15
        rt_2 = 1 / np.sqrt(2)
        # Arm 1 Tasks    # Grab Chassis
        task = np.array([[0 , 0 , 2 , 0.00 , 0.088 , 0.1 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , 0.00 , 0.0735 , 0.009 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 2 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , 1 , -1 , -1 , 0],
                         [0 , 1 , -1 , x_a_p , y_a_p - 0.2 , h_a_p, 0 , 0 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 18 , 0.00 , 0.04 , 0.1 , 0 , 0 , 1 , 0 , 1 , -1 , -1 , 0],
                         [0 , 0 , 18 , 0.00 , 0.04 , 0.054, 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 18 , 0.00 , 0.00 , 0.3 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [4 , 0 , 2 , 0.00 , 0.00 , 0.00 , 1 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                        # Grab PCB
                         [0 , 0 , 9 , 0.00 , 0.00 , 0.030 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 9 , 0.00 , 0.00 , 0.009 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 9 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , x_a_p , y_a_p - 0.2 , h_a_p, 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.0055 , 0.03 , 0.35 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.0055 , 0.03 , 0.0245 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , 1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.0055 , -0.03 , h_a_pl , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , 0.20 , 0.00 , 1.2, 0 , 0 , 1 , 0 , -1 , -1 , -1 , 1]])
                         
        len_task = task.shape[0]
        self._task_array[0 , 0 : len_task , :] = task

        # Arm 2 Tasks     # Grab Battery
        task = np.array([[0 , 0 , 10 , 0.00 , 0.00 , 0.05 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 10 , 0.00 , 0.00 , 0.02 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 10 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , 0.10 , 0.25 , 0.9 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , x_a_p , y_a_p + 0.2 , h_a_p , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , 0.00 , -0.036 , 0.1 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , 0.00 , -0.036 , 0.033 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.008 , -0.035 , h_a_pl , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , 0.00 , 0.3 , h_a_pl , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                        # Grab Cover
                         [0 , 0 , 11 , 0.00 , 0.00 , 0.03 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 11 , 0.00 , 0.00 , 0.01 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 11 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , 0.1 , 0.68 , 1.0 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , 0.1 , y_a_p + 0.1, h_a_p , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.001 , 0.065 , 0.06 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.001 , 0.065 , 0.0475 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 2 , -0.001 , 0.065, 0.0455 , 0 , 0 , 1 , 0 , -1 , 2 , 11 , 0],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         #[4 , 0 , 11 , 0.00 , 0.00 , 0.00 , 1 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                        # Move Cover To Arm 4
                         [0 , 0 , 2 , 0.006 , 0.065, 0.0375 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 11 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.85 , 0.55 , 1.1 , 0 , 0 , 1 , 0 , 3 , -1 , -1 , 1],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , 3 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.7 , 0.7 , 1.2 , 0 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , 0.20 , 0.90 , 1.2, 0 , 0 , 1 , 0 , -1 , -1 , -1 , 1]])
        len_task = task.shape[0]
        self._task_array[1 , 0 : len_task , :] = task

        # Arm 4 Tasks    # Grab Cover Assembly from Arm 2
        task = np.array([[0 , 0 , 11 , -0.06 , 0.00 , 0.0 , 1 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 11 , -0.023 , 0.00 , 0.0 , 1 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , 10 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , 1 , -1 , -1 , 1],
                         [0 , 0 , 17 , -0.022 , 0.05 , 0.20 , 1 , 0 , 1 , 0 , 2 , -1 , -1 , 0],
                         [0 , 0 , 17 , -0.022 , 0.05 , 0.1305 , 1 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 17 , -0.026 , 0.05 , 0.2 , 1 , 0 , 1 , 0 , -1 , -1 , -1 , 0],
                         [4 , 0 , 2 , 0.00 , 0.00 , 0.00 , 1 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                        # Grab Right Side Motor
                         [3 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [3 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 4 , 0.00 , 0.00 , 0.10 , 0 , 1 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 4 , 0.00 , 0.00 , 0.05 , 0 , 1 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 4 , 0.00 , 0.00 , 0.00 , 0 , 1 , 1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , -1 , -1.657 , 0.485 , h_a_p , 0 , 1 , 1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.90 , 0.425 , 0.95 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.94 , y_a_p - 0.1 , h_a_p , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.95 , 0.503 , 1.00 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.95 , 0.503 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.965 , 0.503 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.975 , 0.503 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                        #  [0 , 1 , -1 , -0.985 , 0.497 , 0.883 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                        #  [0 , 1 , -1 , -0.995 , 0.497 , 0.883 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                        #  [0 , 1 , -1 , -0.999 , 0.497 , 0.883 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , 2 , 4 , 0],
                         [3 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 1]])
        len_task = task.shape[0]
        self._task_array[3 , 0 : len_task , :] = task

        # Arm 3 Tasks   # Grab Left Side Motor
        task = np.array([[0 , 0 , 3 , 0.00 , 0.00 , 0.20 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 3 , 0.00 , 0.00 , 0.05 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 0 , 3 , 0.00 , 0.00 , 0.00 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [1 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.90 , 0.425 , 0.95 , 0 , 1 , -1 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.94 , y_a_p - 0.1 , h_a_p , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.17 , 0.503 , 1.00 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.16 , 0.503 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.15 , 0.497 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.14 , 0.497 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -1.13 , 0.497 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [0 , 1 , -1 , -0.121 , 0.497 , 0.87 , 0 , 1 , 0 , 0 , -1 , -1 , -1 , 0],
                         [2 , 0 , -1 , 0.00 , 0.0 , 0.00 , 0 , 0 , 0 , 0 , -1 , 2 , 3 , 0],
                         [3 , 0 , -1 , 0.00 , 0.00 , 0.00 , 0 , 0 , 0 , 0 , -1 , -1 , -1 , 1]])
        len_task = task.shape[0]
        self._task_array[2 , 0 : len_task , :] = task

        self._controllers = []
        return

    def setup_scene(self):
        world = self.get_world()
        # We add the task to the world here
        world.add_task(self._task)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        self._objs = self._task._objs
        self._objs_kit1_second = self._task._objs_kit1_second
        self._objs_kit2_second = self._task._objs_kit2_second
        self._robots = self._task._robots

        for i in range(self._task._num_arms):
            self._controllers.append(
                RMPFlowController(
                    name="RMP_controller",
                    robot_articulation=self._robots[i],
                    physics_dt = 1.0/30.0
                )
            )

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):

        self._task_index = np.array([0,0,0,0])
        self._in_use = np.array([0,0,0,0])
        self._wait_time = np.array([0,0,0,0])
        self._gripping = np.array([0,0,0,0])
        self.is_gripping = np.array([-1,-1,-1,-1])
        self._kit1_ready = 0
        self._kit2_ready = 0
        
        # break fixed joints
        #for i in range(len())
        delete_prim("/World/Obj_11/FixedJoint")
        delete_prim("/World/Obj_2/FixedJoint")
        for i in range(self._task._num_arms):
            self._controllers[i].reset()

        await self._world.play_async()
        return

    def physics_step(self, step_size):

        # Task Index -> [type , goal_type , obj_ind , x , y , z , q1 , q2 , q3 , q4 , start_arm? , f_ob1 , f_obj2 , last_task?]
                        # 0         1          2      3   4   5    6   7    8    9       10         11      12          13

        # if len(np.where(self._in_use == 1)[0]) == 0:
        #     self._world.pause()
        # print(self._in_use)
        for i in range(4): # Checks for each arm

            if self._in_use[i]: # See if arm is in use

                curr_task = self._task_array[ i , self._task_index[i] , : ]

                if curr_task[0] == 0: # Move

                    gripper_location = self._robots[i].end_effector.get_current_dynamic_state().position

                    if curr_task[1] == 0:               # Object Goal
                        #print(np.int16(curr_task[2]), self._objs)
                        goal_position = self._objs[ np.int16(curr_task[2]) ].get_world_pose()[0]
                        goal_position = goal_position + curr_task[3:6]
                    else:                               # Position Goal
                        goal_position = curr_task[3:6]

                    if self._is_gripping[i] != -1:
                        # curr_obj_position = self._objs[self._is_gripping[i]].get_world_pose()[0]
                        # obj_to_arm = gripper_location - curr_obj_position
                        # goal_position = goal_position + obj_to_arm
                        diff_margin = 0.04725
                    else:
                        diff_margin = 0.0585

                    if np.linalg.norm(gripper_location - goal_position) > diff_margin:
                        #print(i, self._in_use , goal_position, curr_task[6:10], self._task_index[i])
                        actions = self._controllers[i].forward(goal_position, curr_task[6:10])
                        self._robots[i].apply_action(actions)
                    else: 
                        self._task_index[i] = self._task_index[i] + 1
                        if curr_task[13]: # Check if arm needs to be turned off
                            self._in_use[i] = 0
                        if curr_task[10] != -1: # Check if other arm needs to be turned on
                                self._in_use[np.int16(curr_task[10])] = 1
                        
                        if (np.int16(curr_task[11]) > -1) and (np.int16(curr_task[12]) > -1): # Need to fuse objects
                            stage = omni.usd.get_context().get_stage()
                            from_prim = stage.GetPrimAtPath(self._objs[ np.int32(curr_task[11]) ].prim_path) 
                            to_prim = stage.GetPrimAtPath(self._objs[ np.int32(curr_task[12]) ].prim_path)
                            utils.createJoint(stage, "Fixed", from_prim, to_prim)

                elif curr_task[0] == 1: # Close Gripper

                    if not self._gripping[i]:
                        self._wait_time[i] = time.time() + 1.4
                        self._gripping[i] = 1
                    else:
                        if self._wait_time[i] - time.time() > 0.01:
                            action = self._robots[i].gripper.forward(action = "close")
                            self._robots[i].apply_action(action)
                        else:
                            self._gripping[i] = 0
                            self._is_gripping[i] = curr_task[2]
                            self._task_index[i] = self._task_index[i] + 1

                            if curr_task[13]: # Check if arm needs to be turned off
                                self._in_use[i] = 0

                            if curr_task[10] != -1: # Check if other arm needs to be turned on
                                self._in_use[np.int16(curr_task[10])] = 1

                elif curr_task[0] == 2: # Open Gripper

                    if not self._gripping[i]:
                        self._wait_time[i] = time.time() + 1.3
                        self._gripping[i] = 1
                    else:
                        if self._wait_time[i] - time.time() > 0.01:
                            action = self._robots[i].gripper.forward(action = "open")
                            self._robots[i].apply_action(action)
                        else:
                            self._gripping[i] = 0
                            self._is_gripping[i] = -1
                            self._task_index[i] = self._task_index[i] + 1

                            if curr_task[13]: # Check if arm needs to be turned off
                                self._in_use[i] = 0

                            if curr_task[10] != -1: # Check if other arm needs to be turned on
                                self._in_use[np.int16(curr_task[10])] = 1

                            if (np.int32(curr_task[11]) > -1) and (np.int32(curr_task[12]) > -1): # Need to fuse objects
                                stage = omni.usd.get_context().get_stage()
                                from_prim = stage.GetPrimAtPath(self._task._prim_paths[ np.int32(curr_task[11]) ]) 
                                to_prim = stage.GetPrimAtPath(self._task._prim_paths[ np.int32(curr_task[12]) ])
                                utils.createJoint(stage, "Fixed", from_prim, to_prim)

                elif curr_task[0] == 3: # Do nothing
                    if not self._gripping[i]:
                        self._wait_time[i] = time.time() + 3.0
                        self._gripping[i] = 1
                    else:
                        if self._wait_time[i] < time.time():

                            self._gripping[i] = 0
                            self._task_index[i] = self._task_index[i] + 1

                            if curr_task[13]: # Check if arm needs to be turned off
                                self._in_use[i] = 0

                            if curr_task[10] != -1: # Check if other arm needs to be turned on
                                self._in_use[np.int16(curr_task[10])] = 1

                else:
                    self._objs[ np.int16(curr_task[2]) ].set_world_pose(orientation = curr_task[6:10])
                    self._task_index[i] = self._task_index[i] + 1

        tray1_y = self._objs[0].get_world_pose()[0][1]
        tray1_y_desired = self._positions[0][1]-0.4
        if self._kit1_ready == 0: #need to deliver a kit
            #moving a kit on conveyor
            
            #print(tray1_y_desired)
            if tray1_y > tray1_y_desired: #check position of tray1
                if self._need_fix1 == 1:
                    self._need_fix1 = 0
                    for obj_ind in [2, 9, 10, 11]: # keep em straight
                        stage = omni.usd.get_context().get_stage()
                        from_prim = stage.GetPrimAtPath(self._objs[0].prim_path) 
                        to_prim = stage.GetPrimAtPath(self._objs[obj_ind].prim_path)
                        utils.createJoint(stage, "Fixed", from_prim, to_prim)
                self._objs[0].set_linear_velocity(np.array([0, -0.15, 0]))
            else:
                self._objs[0].set_linear_velocity(np.array([0, 0, 0]))
                self._kit1_ready = 1
                if self._need_fix1 == 0: #since only need to delete joints once
                    delete_prim("/World/Obj_2/FixedJoint")
                    delete_prim("/World/Obj_9/FixedJoint")
                    delete_prim("/World/Obj_10/FixedJoint")
                    delete_prim("/World/Obj_11/FixedJoint")
                    self._in_use[0] = 1 #make first arm begin moving
                self._need_fix1=1

            self._objs[0].set_world_pose(position=self._objs[0].get_world_pose()[0],orientation=np.array([1,0,0,0]))
        

        #need to set kit1_ready to 0 if all parts taken from it (object 11 is cover which is last to leave)
        cover_pos = self._objs[11].get_world_pose()[0]
        cover_pos_kitted = self._positions[2]-np.array([0,0.4,0])
        if self._kit1_ready == 1 and np.linalg.norm(cover_pos - cover_pos_kitted) > 0.4:
            self._objs[0].set_linear_velocity(np.array([0, -0.15, 0]))
        
        #new kit1 needed and both first arms are not in use
        cover_pos = self._objs[11].get_world_pose()[0]
        tray_pos = self._objs[0].get_world_pose()[0]
        if self._kit1_ready == 1 and sum(self._in_use)==0 and np.linalg.norm(cover_pos - tray_pos) > 0.4:
            for obj_ind in [0,2,9,10,11]:
                #point robots to new kit
                index_in_kit1 = 0
                self._objs[obj_ind] = self._objs_kit1_second[index_in_kit1]
                index_in_kit1 = index_in_kit1 + 1
            self._kit1_ready = 0


        tray2_y = self._objs[1].get_world_pose()[0][1]
        tray2_y_desired = self._positions[1][1]-0.4
        if self._kit2_ready == 0: #need to deliver a kit
            #moving a kit on conveyor
            if tray2_y > tray2_y_desired: #check position of tray1
                if self._need_fix2 == 1:
                    self._need_fix2 = 0
                    for obj_ind in [3,4,5,6,7,8]: # keep em straight
                        stage = omni.usd.get_context().get_stage()
                        from_prim = stage.GetPrimAtPath(self._objs[1].prim_path) 
                        to_prim = stage.GetPrimAtPath(self._objs[obj_ind].prim_path)
                        utils.createJoint(stage, "Fixed", from_prim, to_prim)
                self._objs[1].set_linear_velocity(np.array([0, -0.15, 0]))
            else:
                self._objs[1].set_linear_velocity(np.array([0, 0, 0]))
                
                self._kit2_ready = 1
                if self._need_fix2 == 0: #since only need to delete joints once
                    delete_prim("/World/Obj_3/FixedJoint")
                    delete_prim("/World/Obj_4/FixedJoint")
                    delete_prim("/World/Obj_5/FixedJoint")
                    delete_prim("/World/Obj_6/FixedJoint")
                    delete_prim("/World/Obj_7/FixedJoint")
                    delete_prim("/World/Obj_8/FixedJoint")
                self._need_fix2=1

            self._objs[1].set_world_pose(position=self._objs[1].get_world_pose()[0],orientation=np.array([1,0,0,0]))

        # cover_pos = self._objs[11].get_world_pose()[0]
        # cover_pos_kitted = self._positions[2]-np.array([0,0.4,0])
        # if self._kit2_ready == 1 and np.linalg.norm(cover_pos - cover_pos_kitted) > 0.4:
        #     self._objs[1].set_linear_velocity(np.array([0, -0.15, 0]))

        # cover_pos = self._objs[11].get_world_pose()[0]
        # tray_pos = self._objs[1].get_world_pose()[0]
        # if self._kit2_ready == 1 and sum(self._in_use)==0 and np.linalg.norm(cover_pos - tray_pos) > 0.4:
        #     for obj_ind in [0,2,9,10,11]:
        #         #point robots to new kit
        #         index_in_kit2 = 0
        #         self._objs[obj_ind] = self._objs_kit1_second[index_in_kit2]
        #         index_in_kit2 = index_in_kit2 + 1
        #     self._kit2_ready = 0

        self._in_use_sum = self._in_use_sum + self._in_use
        self._physics_step_counter = self._physics_step_counter + 1
        print(self._in_use_sum / self._physics_step_counter)
        return
