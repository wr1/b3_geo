
= separate 2d and 3d mesh generation workflows
- lm1_mesh.vtp --> 2d --> runs when there is a mesh chapter in the yaml file, serves as input for 2d section mesh gen
- lm2_mesh3d.vtp --> 3d --> runs when there is a mesh3d chapter in the yaml file, serves as input for surface mesh gen

The work to create these is the same in this code (interpolating airfoils at given z locations), but they are stored in separate vtp outputs
