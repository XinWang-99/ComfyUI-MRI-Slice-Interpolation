# ComfyUI-MRI-Slice-Interpolation

This repo provides a custom node of MRI slice interpolation for ComfyUI. Please follow the steps below to set up and use the tool in your environment.

![image](https://github.com/XinWang-99/ComfyUI-MRI-Slice-Interpolation/assets/49643166/03285b87-722b-40f4-8cec-24877e06af60)



## Step 1: Install ComfyUI

First, you need to install ComfyUI. Detailed installation instructions are available at the following link:

[ComfyUI Installation Instructions](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#installing)


## Step 2: Download This Repository

Clone this repository and add it to the ComfyUI environment by placing it under `ComfyUI/custom_nodes`. You can use the following commands:

```bash
git clone https://github.com/XinWang-99/ComfyUI-MRI-Slice-Interpolation.git
mv ComfyUI-MRI-Slice-Interpolation/child_node ComfyUI/custom_nodes/
```
### Step 3: Run the Application
Set the GPU device and run the application using the following command:

```bash
CUDA_VISIBLE_DEVICES=[your gpu id] python main.py
```

### Step 4: Load the Workflow
Once ComfyUI is running, load the workflow.json provided in this repository.
You can upload your own .nii file, choose different super-resolution model and super-resolution ratio, and get a final .nii file.
