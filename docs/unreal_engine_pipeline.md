# Unreal Engine 5 Automated Multi-Modal Data Collection Pipeline

Designed and developed to automate synthetic data generation in **Unreal Engine 5** using the **Unreal Python API (`import unreal`)**.

---

## 1. Pipeline Overview

```
+-------------------------------------------------------------------------+
|                         Unreal Engine 5 Level                           |
|       (Nanite Geometry, Lumen Dynamic GI, Physics Simulation)           |
+-------------------------------------------------------------------------+
                                    |
      +-----------------------------+-----------------------------+
      | (1) Camera Path Generator   | (2) Multi-Pass Renderer     |
      |   - 3D Spherical Orbits     |   - FinalImage (Lit RGB)    |
      |   - Catmull-Rom Splines     |   - SceneDepth (16-bit)     |
      |   - Perlin 6-DOF Flythrough |   - WorldNormal (GBuffer)   |
      +-----------------------------+-----------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Headless Unreal Python Automation                      |
|           UnrealEditor-Cmd.exe -run=pythonscript -script=...            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                       Dataset Directory Structure                       |
|   data/unreal_dataset/                                                  |
|     ├── scene_0001/                                                     |
|     │     ├── rgb/frame_0000.png ... frame_0015.png                     |
|     │     ├── depth/depth_0000.png ... depth_0015.png                   |
|     │     ├── normals/normal_0000.png ... normal_0015.png               |
|     │     └── metadata.json (extrinsics 4x4, intrinsics K)              |
+-------------------------------------------------------------------------+
```

---

## 2. Unreal Engine Configuration

### 2.1 Enabling Required Plugins
In Unreal Engine 5 (`Project Settings -> Plugins`):
1. **Python Editor Script Plugin** (`PythonScriptPlugin`): Enables the in-engine Python 3 interpreter and `unreal` module.
2. **Editor Scripting Utilities** (`EditorScriptingUtilities`): Provides level/actor manipulation methods (`unreal.EditorLevelLibrary`).
3. **Movie Render Queue** (`MovieRenderPipeline`): For multi-pass cinematic rendering.

Enable developer mode in `Edit -> Project Settings -> Plugins -> Python`:
- Enable `Enable Remote Execution`
- Enable `Developer Mode`

---

## 3. Render Passes Explained

### 3.1 2D Projections (Lit RGB)
- Standard deferred/clustered forward pass with Lumen global illumination and physical shadows.
- Output format: 8-bit sRGB PNG per frame.

### 3.2 Depth Maps (`SceneDepth`)
- Represents metric camera-to-surface distance $Z$ in Unreal world units (centimeters).
- Post-process material normalizes distance $d_{norm} = \text{clamp}(Z / Z_{max}, 0, 1)$.
- Exported as 16-bit single-channel grayscale PNG or 32-bit linear EXR.

### 3.3 Surface Normal Maps (`WorldNormal`)
- Captures surface normal unit vectors $\mathbf{N} = (N_x, N_y, N_z) \in [-1, 1]^3$ from the G-Buffer.
- Encoded into 8-bit RGB color space via:
  $$\mathbf{C}_{RGB} = \text{round}\left( 255 \cdot \frac{\mathbf{N} + 1}{2} \right)$$
- Preserves fine surface curvature and high-frequency geometric contours.

### 3.4 Camera Metadata JSON
Contains exact extrinsic and intrinsic matrices per frame:
- **Intrinsics $K$**:
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad f_x = \frac{W / 2}{\tan(\text{FOV} / 2)}$$
- **Extrinsics**: 6-DOF camera position $(X, Y, Z)$, Euler rotation angles (Pitch, Yaw, Roll), and orientation Quaternions $(q_x, q_y, q_z, q_w)$.

---

## 4. Headless Execution

To run automated data collection without opening the Unreal Editor GUI:

```powershell
& "C:\Program Files\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
  "C:\UnrealProjects\GenerativeDiffusionSim\GenerativeDiffusionSim.uproject" `
  -run=pythonscript `
  -script="C:\path\to\GenerativeDiffusionModelforVideo\src\unreal\ue_python_capture.py" `
  -unattended -nopause -nosplash -stdout
```

Or execute via the Python orchestrator:
```python
from src.unreal.ue_runner import UnrealHeadlessRunner

runner = UnrealHeadlessRunner(project_uproject="C:/Projects/GenerativeDiffusion.uproject")
runner.run_script("src/unreal/ue_python_capture.py")
```

