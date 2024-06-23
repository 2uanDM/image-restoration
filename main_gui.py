import os
import shutil
import subprocess
import sys
import gradio as gr
from gradio_imageslider import ImageSlider

root_dir = "/home/quan/workspace/image-restoration"

from PIL import Image
theme = gr.themes.Default(primary_hue="blue").set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_300",
)

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

def run_cmd(command):
    try:
        subprocess.call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def reset():
    return (None)

def fn(input_image):
    return input_image


def update_output_image(input_image): 
    input_image = input_image[0]
    # Rebuild the .temp folder
    shutil.rmtree(os.path.join(root_dir, "Global", ".temp"), ignore_errors=True)
    os.makedirs(os.path.join(root_dir, "Global", ".temp"), exist_ok=True)
    
    # Temporarily save the input image
    input_image_dir = os.path.join(root_dir, "Global", ".temp", "input_image")
    os.makedirs(input_image_dir, exist_ok=True)
    # image = Image.fromarray(input_image)
    input_image.save(os.path.join(input_image_dir, "input_image.jpg"))
    
    
    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_output_dir = os.path.join(".temp", "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    
    # stage_1_command = (
    #     "python test.py --test_mode Full --Quality_restore --test_input "
    #     + input_image_dir
    #     + " --outputs_dir "
    #     + stage_1_output_dir
    #     + " --gpu_ids "
    #     + "0"
    # )
    # run_cmd(stage_1_command)
    
    mask_dir = os.path.join(stage_1_output_dir, "masks")
    new_input = os.path.join(mask_dir, "input")
    new_mask = os.path.join(mask_dir, "mask")
    stage_1_command_1 = (
        "python detection.py --test_path "
        + input_image_dir
        + " --output_dir "
        + mask_dir
        + " --input_size full_size"
        + " --GPU "
        + "0"
    )

    stage_1_command_2 = (
        "python test.py --Scratch_and_Quality_restore --test_input "
        + new_input
        + " --test_mask "
        + new_mask
        + " --outputs_dir "
        + stage_1_output_dir
        + " --gpu_ids "
        + "0"
    )

    run_cmd(stage_1_command_1)
    run_cmd(stage_1_command_2)

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(os.path.join(".temp", "final_output"))
    os.makedirs(stage_4_output_dir, exist_ok=True)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...")
    print("\n")
    print("All the processing is done. Please check the results.")
    
    # Read the output image:
    output_image = Image.open(os.path.join(".temp/stage_1_restore_output/restored_image/input_image.png"))
    height, width = output_image.size
    os.chdir(root_dir)
    # Return the image and its dimensions
    return input_image, output_image
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate"),
    fill_height=True,
) as demo:
    gr.Markdown("## Image Restoration Demo")
    
    img1 = ImageSlider(label="Blur image", type="pil")
    with gr.Row():
        restore_button = gr.Button("Restore Image")
        reset_button = gr.Button("Reset Image")
    restore_button.click(update_output_image, inputs=img1, outputs=img1)
    reset_button.click(reset, outputs=[img1])

    
        
demo.launch()
