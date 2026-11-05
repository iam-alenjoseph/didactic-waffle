# Spatio-Temporal Video UNet Architecture Diagram

```
+---------------------------------------------------------------------------------------+
|                                    INPUT VIDEO TENSOR                                  |
|                                       x_t: (B, 3, T, H, W)                            |
|                            Condition c = {Depth, Normal}: (B, 4, T, H, W)             |
+---------------------------------------------------------------------------------------+
                                            |
                                            v
                                  [ Channel Concatenation ]
                                  Input: (B, 7, T, H, W)
                                            |
                                            v
                                  [ Conv3D (1x3x3) Pre-Proj ]
                                            |
                                            v
+==================================== ENCODER (DOWNSAMPLE) ==============================+
| Level 1: [ResNet3D (1x3x3 + 3x1x1)] -> [Temporal Self-Attn] -> [Downsample3D (1x2x2)]  |
| Level 2: [ResNet3D (1x3x3 + 3x1x1)] -> [Temporal Self-Attn] -> [Downsample3D (1x2x2)]  |
| Level 3: [ResNet3D (1x3x3 + 3x1x1)] -> [Temporal Self-Attn] -> [Downsample3D (1x2x2)]  |
+========================================================================================+
                                            |
                                            v
+======================================= BOTTLENECK =====================================+
|   [ResNet3D Block]  <--- Timestep Embedding AdaGroupNorm(scale, shift)                |
|   [Temporal Self-Attention across T dimension]                                        |
|   [Cross-Attention with Condition Tokens]                                             |
|   [ResNet3D Block]                                                                    |
+========================================================================================+
                                            |
                                            v
+==================================== DECODER (UPSAMPLE) ================================+
| Level 3: [Upsample3D (1x2x2)] -> [Concat Skip] -> [ResNet3D] -> [Temporal Self-Attn]   |
| Level 2: [Upsample3D (1x2x2)] -> [Concat Skip] -> [ResNet3D] -> [Temporal Self-Attn]   |
| Level 1: [Upsample3D (1x2x2)] -> [Concat Skip] -> [ResNet3D] -> [Temporal Self-Attn]   |
+========================================================================================+
                                            |
                                            v
                                  [ Final GroupNorm & SiLU ]
                                  [ Conv3D (1x3x3) Out-Proj ]
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                  OUTPUT PREDICTION                                    |
|                       epsilon_theta(x_t, t, c): (B, 3, T, H, W)                       |
+---------------------------------------------------------------------------------------+
```

