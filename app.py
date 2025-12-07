import streamlit as st
import os
import sys
import json
import torch
from PIL import Image
import numpy as np
import re
import ast

# Add project root and subdirs to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agent')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'distance_est')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'inside_pred')))

# Import tools and agent logic
try:
    from agent.tools import tools_api
    from agent.mask import parse_masks_from_conversation
    from google import genai
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure all dependencies are installed and directories are correct.")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="Spatial Agent (Local)", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    model_name = st.selectbox("Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-exp-1206"], index=0)
    
    st.divider()
    st.header("Data Source")
    data_path = "data/val"
    json_file = st.file_uploader("Upload Annotations JSON (e.g., rephrased_val.json)", type=['json'])
    
    st.divider()
    st.header("Model Config")
    # Default paths based on repo structure
    dist_ckpt = st.text_input("Distance Model Ckpt", "distance_est/ckpt/epoch_5_iter_6831.pth")
    inside_ckpt = st.text_input("Inside Model Ckpt", "inside_pred/ckpt/epoch_4.pth")
    small_dist_ckpt = st.text_input("Small Dist Model Ckpt", "distance_est/ckpt/3m_epoch6.pth")

# --- Helper Functions ---
@st.cache_resource
def load_tools(dist_ckpt, inside_ckpt, small_dist_ckpt):
    try:
        tools = tools_api(
            dist_model_cfg={'model_path': dist_ckpt}, 
            inside_model_cfg={'model_path': inside_ckpt},
            small_dist_model_cfg={'model_path': small_dist_ckpt},
            resize=(360, 640),
            mask_IoU_thres=0.3, inside_thres=0.5,
            cascade_dist_thres=300, clamp_distance_thres=25
        )
        return tools
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def get_image_files(directory):
    img_dir = os.path.join(directory, "images")
    if not os.path.exists(img_dir):
        return []
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# --- Main UI ---
st.title("Spatial Agent Playground")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
    st.stop()

# Load Tools
with st.spinner("Loading models..."):
    tools = load_tools(dist_ckpt, inside_ckpt, small_dist_ckpt)
    if not tools:
        st.stop()

# Load Data
image_files = get_image_files(data_path)
if not image_files:
    st.error(f"No images found in {data_path}/images")
    st.stop()

# Selection
col1, col2 = st.columns([1, 2])

with col1:
    selected_image_file = st.selectbox("Select Image", image_files)
    image_path = os.path.join(data_path, "images", selected_image_file)
    
    # Display Image
    image = Image.open(image_path)
    st.image(image, caption=selected_image_file, use_container_width=True)

    # Find corresponding data in JSON if available
    selected_item = None
    if json_file:
        try:
            data = json.load(json_file)
            # Assuming data is a list of items, find the one matching the image filename
            # The 'image' field in json usually contains the filename
            for item in data:
                if item.get('image') == selected_image_file:
                    selected_item = item
                    break
            
            if selected_item:
                st.success(f"Annotations found! Loaded {len(data)} items.")
                with st.expander("View Annotations"):
                    st.json(selected_item)
            else:
                st.warning("No annotations found for this image in the uploaded JSON.")
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

# Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Agent Session if needed
if "genai_client" not in st.session_state:
    st.session_state.genai_client = genai.Client(vertexai=False, api_key=api_key)

if "agent_chat" not in st.session_state or st.session_state.get("current_model") != model_name:
    st.session_state.agent_chat = st.session_state.genai_client.chats.create(
        model=model_name,
        config=genai.types.GenerateContentConfig(temperature=0.2)
    )
    st.session_state.current_model = model_name
    # Preload prompts
    try:
        with open('agent/prompt/agent_example.txt', 'r') as f:
            st.session_state.prompt_preamble = f.read()
        with open('agent/prompt/answer.txt', 'r') as f:
            st.session_state.answer_preamble = f.read()
    except:
        st.session_state.prompt_preamble = "You are a spatial agent."
        st.session_state.answer_preamble = "Please provide the final answer."

# Display Chat
with col2:
    st.subheader("Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent Execution Loop
        with st.spinner("Agent is thinking..."):
            try:
                # Update tools with current image and masks
                tools.update_image(image_path)
                
                if selected_item:
                    # Parse masks
                    conversation_context = selected_item.get('rephrase_conversations', [{}])[0].get('value', '')
                    rle_data = selected_item.get('rle', [])
                    masks = parse_masks_from_conversation(conversation_context, rle_data)
                    tools.update_masks(masks)
                    with st.expander("Loaded Masks", expanded=False):
                        st.write(list(masks.keys()))
                else:
                    # If no masks, tools won't work for spatial queries requiring them
                    tools.update_masks({})
                
                # Prepare prompt
                # If it's the first message, prepend the preamble
                current_history = [m for m in st.session_state.messages if m["role"] != "system"]
                
                # We construct the message to send to Gemini
                # Ideally we should maintain the chat object state, but for Streamlit we might need to send history
                # or just the new message if the chat object persists.
                # Since we persist agent_chat in session_state, we can just send the message.
                
                # However, for the very first turn, we might want to inject the system prompt/preamble
                if len(st.session_state.messages) == 1:
                     # Inject preamble with available masks
                     masks_info = f"\n\nAvailable masks in this image: {', '.join(list(tools.masks.keys()))}\n"
                     full_prompt = st.session_state.prompt_preamble.replace("<question>", masks_info + prompt)
                     response = st.session_state.agent_chat.send_message(full_prompt)
                else:
                     response = st.session_state.agent_chat.send_message(prompt)
                
                assistant_text = response.text.strip()
                
                # Tool Execution Loop (matching agent_run.py logic)
                max_steps = 10
                step = 0
                final_answer = None
                execute_flag = False  # Track if we've executed at least one tool
                
                while step < max_steps:
                    step += 1
                    
                    # Check for <execute> command
                    execute_match = re.search(r"<execute>(.*?)</execute>", assistant_text, re.DOTALL)
                    if execute_match:
                        execute_flag = True
                        command = execute_match.group(1).strip()
                        st.markdown(f"*Executing: `{command}`*")
                        
                        # Execute Tool
                        try:
                            # Parse command
                            cmd_clean = command.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('<', '').replace('>', '')
                            match = re.match(r"(\w+)\s*\((.*)\)", cmd_clean)
                            if match:
                                func_name, args_str = match.groups()
                                
                                # Map functions
                                func_map = {
                                    "dist": tools.dist,
                                    "closest": tools.closest,
                                    "is_left": tools.is_left,
                                    "is_right": tools.is_right,
                                    "inside": tools.inside,
                                    "most_right": tools.most_right,
                                    "most_left": tools.most_left,
                                    "middle": tools.middle,
                                    "is_empty": tools.is_empty
                                }
                                
                                if func_name in func_map:
                                    # Argument parsing logic
                                    # Define which functions expect a list as their argument
                                    list_arg_funcs = {"most_left", "most_right", "middle", "is_empty"}
                                    list_second_arg_funcs = {"closest", "inside"}  # First arg single, second arg is list
                                    
                                    def resolve_arg(arg_str):
                                        arg_str = arg_str.strip()
                                        # Handle angle brackets if present
                                        if arg_str.startswith('<') and arg_str.endswith('>'):
                                            arg_str = arg_str[1:-1]
                                            
                                        if arg_str.startswith("'") or arg_str.startswith('"'):
                                            return ast.literal_eval(arg_str)
                                        
                                        # Check if it's in the masks dictionary (exact match)
                                        if arg_str in tools.masks:
                                            return tools.masks[arg_str]
                                        
                                        # Check if it's a collective reference (e.g., "pallets", "transporters")
                                        # These should match multiple masks of the same class
                                        # Try singular form first, then check if arg is plural
                                        possible_class = arg_str.rstrip('s')  # Remove trailing 's' for plural
                                        matching_masks = [m for name, m in tools.masks.items() if m.object_class.lower() == possible_class.lower() or m.object_class.lower() == arg_str.lower()]
                                        
                                        if matching_masks:
                                            # Return the list of matching masks
                                            return matching_masks
                                        
                                        # Check if it looks like a mask name pattern (e.g., pallet_0, transporter_1)
                                        # Valid mask names should match: word_number (e.g., pallet_0, buffer_1)
                                        if re.match(r'^[a-zA-Z]+_\d+$', arg_str):
                                            raise ValueError(f"Mask '{arg_str}' not found in loaded masks. Available masks: {list(tools.masks.keys())}")
                                        
                                        # Check if it looks like a placeholder (e.g., pallet_X, transporter_Y)
                                        if re.match(r'^[a-zA-Z]+_[a-zA-Z]+$', arg_str):
                                            raise ValueError(f"Mask placeholder '{arg_str}' detected. Please use actual mask names from: {list(tools.masks.keys())}")
                                        
                                        # Otherwise treat as a string literal
                                        return arg_str

                                    # Split args by comma, respecting bracket nesting
                                    def split_args_respecting_brackets(args_str):
                                        """Split by comma but respect bracket nesting"""
                                        args = []
                                        current_arg = []
                                        bracket_depth = 0
                                        
                                        for char in args_str:
                                            if char == '[':
                                                bracket_depth += 1
                                                current_arg.append(char)
                                            elif char == ']':
                                                bracket_depth -= 1
                                                current_arg.append(char)
                                            elif char == ',' and bracket_depth == 0:
                                                # Comma at top level - split here
                                                args.append(''.join(current_arg).strip())
                                                current_arg = []
                                            else:
                                                current_arg.append(char)
                                        
                                        # Add the last argument
                                        if current_arg:
                                            args.append(''.join(current_arg).strip())
                                        
                                        return args
                                    
                                    # Use the bracket-aware splitter
                                    args_list = split_args_respecting_brackets(args_str)
                                    resolved_args = []
                                    
                                    for arg in args_list:
                                        # Check if this arg is a bracketed list
                                        if arg.startswith('[') and arg.endswith(']'):
                                            # Extract inner content and resolve each item
                                            inner_content = arg[1:-1]
                                            inner_args = [x.strip() for x in inner_content.split(',')]
                                            resolved_args.append([resolve_arg(x) for x in inner_args])
                                        else:
                                            # Regular argument
                                            resolved_args.append(resolve_arg(arg))
                                    
                                    # Now handle functions that expect lists
                                    if func_name in list_arg_funcs:
                                        # All args should be wrapped in a single list
                                        # Unless they're already in a list
                                        if len(resolved_args) == 1 and isinstance(resolved_args[0], list):
                                            # Already a list, use as is
                                            pass
                                        else:
                                            # Wrap all args in a list
                                            resolved_args = [resolved_args]
                                    
                                    elif func_name in list_second_arg_funcs:
                                        # First arg is single, rest should be wrapped in a list
                                        if len(resolved_args) > 1:
                                            if isinstance(resolved_args[1], list):
                                                # Second arg already a list
                                                resolved_args = [resolved_args[0], resolved_args[1]]
                                            else:
                                                # Wrap second and subsequent args in a list
                                                resolved_args = [resolved_args[0], resolved_args[1:]]
                                    
                                    # Debug: Show what we're passing to the function
                                    debug_info = []
                                    for i, arg in enumerate(resolved_args):
                                        if isinstance(arg, list):
                                            list_types = [type(item).__name__ for item in arg]
                                            debug_info.append(f"arg{i}: [{', '.join(list_types)}]")
                                        else:
                                            arg_type = type(arg).__name__
                                            debug_info.append(f"arg{i}: {arg_type}")
                                    st.markdown(f"*Debug: {', '.join(debug_info)}*")
                                    
                                    result = func_map[func_name](*resolved_args)
                                    st.markdown(f"*Result: {result}*")
                                    
                                    # Send result back to agent and continue loop
                                    response = st.session_state.agent_chat.send_message(f"{result}")
                                    assistant_text = response.text.strip()
                                    continue  # Continue the loop to process the new response
                                else:
                                    raise ValueError(f"Unknown function: {func_name}")
                            else:
                                raise ValueError("Invalid command format")
                        except Exception as e:
                            st.error(f"Tool Execution Error: {e}")
                            break
                    
                    # Check for <answer> tag - only accept if execute_flag is True
                    answer_match = re.search(r"<answer>(.*?)</answer>", assistant_text, re.DOTALL)
                    if answer_match and execute_flag:
                        raw_answer = answer_match.group(1).strip()
                        
                        # Format the answer using answer_preamble (like agent_run.py does)
                        format_response = st.session_state.agent_chat.send_message(st.session_state.answer_preamble)
                        final_answer = format_response.text.strip()
                        
                        # Check if the answer is a mask name and convert to region_id
                        if final_answer in tools.masks:
                            final_answer = str(tools.masks[final_answer].region_id)
                        
                        break
                    
                    # If neither execute nor answer, something went wrong
                    if not execute_match and not answer_match:
                        st.warning("Agent response did not contain execute or answer tags.")
                        final_answer = assistant_text
                        break
                
                if final_answer:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                else:
                    st.warning("Agent did not provide a final answer.")

            except Exception as e:
                st.error(f"Error: {e}")
