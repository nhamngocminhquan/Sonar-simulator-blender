# Sonar-simulator-blender ver 0.3
Sonar simulator based on blender ver 0.3. 
# Blender version
Currently support Blender 2.79 and 2.80.  
Blender 2.79 has a rectangular-shaped spot light in Blender renderer which is similar to sonar wave emission; however, the accuracy and precision of Blender renderer cannot be guaranteed. For other rendered, using Blenderer 2.80 would be much better, which has more functions. 
# Dependencies
Opencv and OpenEXR.  
For Opencv in Windows, don't forget to copy dll file to your workspace. To link Opencv to blender, try  
```
import sys  
sys.path.append(path to the correct Python version in Opencv)  
```
Although, the program is cross platform, the uploaded OpenEXR only supports windows.  
Installation for Windows can follow:  
https://www.kunihikokaneko.com/dblab/cg/bpypip.html  
You can install everything you need in Blender.  
# Tasks
- [x] Raw image generation  
- [x] Fan-shaped image generation will be available soon
- [x] Underwater scene rendering
- [ ] Combination of sonar simulation and underwater scene rendering
- [ ] Noise and distortion simulation  
- [ ] Anti-aliasing  

# Tips
1. The Python script to generate acoustic images is written in "Sonar image simulator" in Text editor in Blender 2.80.  
2. Modifying the configuration of camera and objects can generate different images. With Blender's powerful GUI, a lot of scenes can be simulated easily.  
3. You can also try adding objects with different materials by defining different material characteristics.
# Related papers
1. Kwak et al., Development of acoustic camera-imaging simulator based on novel model, 2015  
2. Mai et al., Acoustic Image Simulator Based on Active Sonar Model in Underwater Environment, 2018  
3. Cerqueira et al.,  A novel GPU-based sonar simulator for real-time applications, 2017       
4. Wook et al., Physically based Sonar Simulation and Image Generation for Side-scan Sonar, 2017
