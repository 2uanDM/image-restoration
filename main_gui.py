import os
import shutil
import sys
import gradio as gr
import subprocess
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

def update_output_image(scratch_opts, input_image):
    # Get the scratch option
    with_scratch = scratch_opts == "With Scratch"
    
    # Get input image dimensions
    height, width = input_image.shape[:2]
    
    # Rebuild the .temp folder
    shutil.rmtree(".temp", ignore_errors=True)
    os.makedirs(".temp", exist_ok=True)
    
    # Temporarily save the input image
    input_image_dir = os.path.join(".temp", "input_image")
    os.makedirs(input_image_dir, exist_ok=True)
    image = Image.fromarray(input_image)
    image.save(os.path.join(input_image_dir, "input_image.jpg"))
    
    print(os.getcwd())
    os.chdir("~/workspace/image-restoration")
    
    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_output_dir = os.path.join(".temp", "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    if not with_scratch:
        stage_1_command = (
            "python test.py --test_mode Full --Quality_restore --test_input "
            + input_image_dir
            + " --outputs_dir "
            + stage_1_output_dir
            + " --gpu_ids "
            + "0"
        )
        run_cmd(stage_1_command)
    else:
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
    stage_4_output_dir = os.path.join(".temp", "final_output")
    os.makedirs(stage_4_output_dir, exist_ok=True)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...")
    print("\n")

    # Stage 2: Face Detection

    # print("Running Stage 2: Face Detection")
    # stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    # stage_2_output_dir = os.path.join(
    #     ".temp", "stage_2_detection_output")
    # os.makedirs(stage_2_output_dir, exist_ok=True)
    # stage_2_command = (
    #     "python Face_Detection/detect_all_dlib.py --url " + stage_2_input_dir +
    #     " --save_url " + stage_2_output_dir
    # )
    # run_cmd(stage_2_command)
    # print("Finish Stage 2 ...")
    # print("\n")

    # # Stage 3: Face Restore
    # print("Running Stage 3: Face Enhancement")
    # stage_3_input_mask = "Face_Enhancement"
    # stage_3_input_face = stage_2_output_dir
    # stage_3_output_dir = os.path.join(
    #     ".temp", "stage_3_face_output")
    # os.makedirs(stage_3_output_dir, exist_ok=True)
    # stage_3_command = (
    #     "python Face_Enhancement/test_face.py --old_face_folder "
    #     + stage_3_input_face
    #     + " --old_face_label_folder "
    #     + stage_3_input_mask
    #     + " --gpu_ids "
    #     + "0"
    #     + " --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir "
    #     + stage_3_output_dir
    #     + " --no_parsing_map"
    # )
    # run_cmd(stage_3_command)
    # print("Finish Stage 3 ...")
    # print("\n")

    # # Stage 4: Warp back
    # print("Running Stage 4: Blending")
    # stage_4_input_image_dir = os.path.join(
    #     stage_1_output_dir, "restored_image")
    # stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    # stage_4_output_dir = os.path.join(".temp", "final_output")
    # os.makedirs(stage_4_output_dir, exist_ok=True)
    # stage_4_command = (
    #     "python Face_Detection/align_warp_back_multiple_dlib.py --origin_url "
    #     + stage_4_input_image_dir
    #     + " --replace_url "
    #     + stage_4_input_face_dir
    #     + " --save_url "
    #     + stage_4_output_dir
    # )
    # run_cmd(stage_4_command)
    # print("Finish Stage 4 ...")
    # print("\n")

    print("All the processing is done. Please check the results.")
    
    # Read the output image:
    output_image = Image.open(".temp/stage_1_restore_output/restored_image/input_image.png")
    
    # Return the image and its dimensions
    return output_image, gr.update(height=height, width=width)

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate"),
    js=js_func,
    fill_height=True,
) as demo:
    gr.Markdown("## Image Restoration Demo")
    
    with gr.Row():
        with gr.Column() as col1:
            input_image = gr.Image(label="Input Image")
            submit_button = gr.Button("Restore Image")
        
        with gr.Column() as col2:
            radio = gr.Radio(
                choices=["With Scratch", "Without Scratch"],
                label="Restoration Mode"
            )
        
    with gr.Column() as col:
        output_image = gr.Image(label="Output Image")

    # Register the function to the button
    submit_button.click(
        update_output_image,
        inputs=[radio, input_image],
        outputs=[output_image, output_image],
    )
        
demo.launch()
