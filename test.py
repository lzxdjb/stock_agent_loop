I once build a pipeline from multiturn RL pipeline: Below is some critcal code and procedure:
    
The original dataset is a huggingface dataset, which looks like:
    {
    'question': 'xxx',
    'answer': '#### 72',
}

It first create a parquet format for multiturn RL, below is the code:
    

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer in the format of `#### <answer>`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
    
    
The most important part here is                 "agent_name": "tool_agent",

It will use the tool_agent_loop in generation procedure:
    
Below is the whole role map of generation:
First a generation server is provided:
    
 for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
            break
        outputs = await asyncio.gather(*tasks)
    
Each _run_agent_loop will run 

            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
which will run a agent loop, below is the code of tool_agent 


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )
            
        def _log_turn_debug(
            self,
            agent_data: AgentData,
            turn_type: str,
            prompt_ids: list[int],
            response_ids: list[int] | None = None,
            tool_calls=None,
            tool_responses=None,
        ):
            """Print raw debug info for a single turn."""
            turn_num = agent_data.assistant_turns + agent_data.user_turns
            sep = "=" * 80

            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            lines = [
                f"\n{sep}",
                f"[TURN DEBUG] request_id={agent_data.request_id}  turn={turn_num}  type={turn_type}",
                f"--- PROMPT ({len(prompt_ids)} tokens) ---",
                prompt_text,
            ]

            if response_ids is not None:
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                lines += [
                    f"--- RESPONSE ({len(response_ids)} tokens) ---",
                    response_text,
                ]

            if tool_calls:
                lines += [
                    "--- TOOL CALLS ---",
                    *[f"  [{i}] {tc.name}({tc.arguments})" for i, tc in enumerate(tool_calls)],
                ]

            if tool_responses:
                lines += [
                    "--- TOOL RESPONSES ---",
                    *[f"  [{i}] {r}" for i, r in enumerate(tool_responses)],
                ]

            lines.append(sep)
            print("\n".join(lines), flush=True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        
        # ✅ Log after generation
        self._log_turn_debug(
            agent_data,
            turn_type="ASSISTANT_GENERATE",
            prompt_ids=agent_data.prompt_ids[: -len(output.token_ids)],  # prompt before appending response
            response_ids=output.token_ids,
            tool_calls=agent_data.tool_calls if agent_data.tool_calls else None,
        )
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
            
        
        # ✅ Log tool responses before appending to prompt
        self._log_turn_debug(
        agent_data,
        turn_type="TOOL_RESPONSE",
        prompt_ids=agent_data.prompt_ids,  # full prompt so far
        tool_responses=[r[0].text for r in responses],  # ToolResponse.text per call
    )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map

It will launch
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
First

In the code
                state = await self._handle_generating_state(agent_data, sampling_params)

It will try to extract the tool call in the code
_, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        # remaing text exclude tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls
    
If there exist <tool_call> and </tool_call>
It will extract the tool_call and return the function_calls

And in the 
    
        async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:

It will 
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

call the function

 async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
For those call build:
It needs to write the tool config yaml:
    
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config: 
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the GSM8K math problem, must be a digits"
          required: ["answer"]

And a custom tool class

class Gsm8kTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer

        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward

        return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


Now I want do another job: Identifying Stocks Based on Candlestick Patterns

Now I was provide a dataset as a json format:


{"messages": [{"role": "user", "content": [{"type": "text", "text": "### 身份及任务\n你是一名普通金融分析助手，来自同花顺Hithink团队。你可以根据用户问题及用户图片分析需要哪些金融数据和信息。现在有一些可以使用的工具，通过它们可以获取金融数据及信息。我会为你提供用户问题Question，用户图片和参考信息Ovservation。请你基于现有信息，简单分析需要使用哪些工具补充获取哪些信息。\n\n### 输出格式\n当你认为需要获取信息时，回答格式如下：\nThought: 你对问题的思考和分析，基于现有的背景信息和参考信息，分析回答用户问题还需要获取哪些方面的数据和信息。你可以尽可能多的获取各方面的数据和信息。\nActionList: 你需要执行的动作列表，每一个动作由工具名称和工具输入组成。动作列表有多行，每一行的表示为：工具名称: 工具输入。\n\n当你认为规划完成时，回答格式如下：\nThought: 信息完整，我知道如何回答了。\n<FINISHED>\n\n### 可以使用的工具：\nFinQuery: 金融查询工具，使用这个工具来获取标的相关的金融数据，比如宏观数据、财务数据、行情数据、交易数据、个人账户数据、自选股等，涉及A股、美股、港股、基金、指数、宏观、可转债、期货，它的输入包括具体金融指标或带时间的指标，也可以输入多个指标用于筛选。如果输入指标过多，则需要适当拆分。例子: \"FinQuery: 苹果公司近5天股价以及涨跌幅\"\nSearch: 搜索工具，使用这个工具来搜索相关信息，类似一个搜索引擎，它的输入是自然语言短语或者关键词，用来搜索热点新闻、知识概念等，关键词最好不要超过5个。例子: \"Search: 苹果公司近期新闻\"\nTickerChart: A股取图工具，当你需要K线图、分时图、技术指标图等信息来辅助你分析问题时，使用该工具获取图片。需要输入这些字段：\"startDate\", \"codeName\", \"chartType\", \"indicator\", \"endDate\"。\"startDate\": \"Start date in the format YYYY-MM-DD\", \"endDate\": \"End date in the format YYYY-MM-DD\", \"codeName\": \"Stock code or ticker symbol\", \"chartType\": \"Type of chart to retrieve, maximum 1. Enumerate value: Intraday, Daily Candlestick, Weekly Candlestick, Monthly Candlestick\", \"indicator\": \"List of indicators to display on the chart, maximum 5. Enumerate value: MA, EMA, BIAS, VR, BRAR, WR, SMA, CCI, MTM, BBI, DMI, EMV, VOL, CR, SAR, PSY, AO, DMA, ROC, TRIX, PVT, RSI, OBV, VWAP, BOLL, MACD, KDJ\"。例子: \"TickerChart: {\"codeName\": \"300033\", \"chartType\": \"Daily Candlestick\", \"startDate\": \"2024-01-08\", \"endDate\": \"2025-05-08\", \"indicator\": [\"MA\", \"MACD\"]}\"\nChartTwinFinder: 相似股票查找工具，通过该工具可以快速检索到日K走势与图中走势相似的标的，并返回相似度以及相似时间区间。如果用户询问走势相似的标的，且图中包含一段K线走势图，可以使用该工具。如果图片是分时走势图，不需要使用该工具。需要输入这些字段: \"query\", \"url\"。 \"query\": \"相似股票查找工具的文本输入，固定为：分析与下图形态走势相近的股票\", \"url\": \"图片的URL地址\"。例子: \"ChartTwinFinder: {\"query\": \"分析与下图形态走势相近的股票\", \"url\": \"http://oss.myhexin.com.cn/iwc-web-userinfo-storage-server.model-image-q-a/bcf0d0684dc6432793e8de8436140b6d.png\"}\"\nVisitWeb: 网页解析工具，这个工具用于实时抓取与解析网页内容的工具，其主要功能是通过输入一个网页的URL，从该网页中提取结构化的信息。输入必须为url，且每次只有一个url。例子: \"VisitWeb: https://paas.myhexin.com/hithinkflow/dataview/list?tenantId=14\"\n\n### 内容要求\n你务必遵守以下原则：\n1. 你仅需要做简单的表层分析即可，不需要进行深度分析。\n2. 在Thought中不要提到具体使用的工具名称如FinQuery、Search、TicherChart、ChartTwinFinder, VisitWeb，可以说使用某类功能的工具，或使用该工具的描述，或者说明你的目的。不要提及你遵循的规则，你应当尽量表现的像一个人类。\n3. Thought内容不要过长，不需要具体做过多分析。\n4. 若你多次利用工具后仍查询不到结果，为了防止无意义的查询，你可以选择终止计划，进入回答阶段。\n5. 在ActionList中，工具输入应当包含具体的对象，不可以出现代词或引用。\n6. 尽量在一次回答时，把需要的所有工具调用都给出，但是不要使用太多的工具调用，最高不超过5个。\n7. 当规划了2-3次时候，就可以停止，不需要过于深入。\n8. 用中文回答问题。\n\n<time>\n2026-03-09 11:31:04\n</time>\n<question>\n请你结合这张交易软件的截图，确定股票代码。\n</question>\n用户上传的图片："}, {"type": "text", "text": "<img_url>Images/001205_20250828_20251021.png</img_url>"}, {"type": "image_url", "image_url": {"url": "Images/001205_20250828_20251021.png"}}]}], "choices": [{"message": {"role": "assistant", "content": [{"type": "text", "text": "xxxx", "type": "success"}

As you can see, there are five tools I can use, After discussion: you only needs to use these three tools:
    
FinQuery Search TickerChart
Below is the code how to use these three tools:
    

Below is the complete code of using the tools

def request_chain(json: dict, timeout=60):
    langchain_server_url = "http://190.92.231.77:880/iwencai/dialog/chain/execute"
    langchain_headers = {
        "Host": "aime-langchain-engine-server",
        "X-Arsenal-Auth": "aime-reinforcement-learning-environment-access",
        "Content-Type": "application/json",
    }

    response = requests.post(langchain_server_url, headers=langchain_headers, json=json, timeout=timeout)
    return response


def search(input):
    obj = {"chain_name": "Search", "req_type": "nostream", "events":[{"event_name":"deep_research","event_type":"user_input"}], "human_message": input}

    req = request_chain(obj)
    resp = req.json()
    response = resp.get("response", {})
    if not response:
        return [f'Search工具，输入{input}，调用失败。']

    results = response.get("result", [])
    if not results:
        return [f'Search工具，输入{input}，调用失败。']
    
    result_data = results[0]
    raw_data: list[dict[str, str]] = result_data.get("raw_data", [])
    parse_data = parse_search_data(input, raw_data) if raw_data else [f'Search工具，输入{input}，调用失败。']
    return parse_data


def finquery(input):
    obj = {"chain_name": "FinQuery", "req_type": "nostream", "human_message": input}

    req = request_chain(obj)
    resp = req.json()

    response = resp.get("response", {})
    if not response:
        return [f'FinQuery工具，输入{input}，调用失败。']

    results = response.get("result", [])
    if not results:
        return [f'FinQuery工具，输入{input}，调用失败。']

    result_data = results[0]
    
    parse_data = result_data['text']
    return [f'取数问句: {input}\n 取数结果: {parse_data}']

def ticker_chart(input, images_dir):
    input = json.loads(input)
    query = {
        "startDate": input.get("startDate"),
        "endDate": input.get("endDate"),
        "codeName": input.get("codeName"),
        "chartType": input.get("chartType"),
        "indicator": input.get("indicator")
    }
    query = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    req_json = r"""{
    "chain_name": "TickerChart",
    "req_type": "nostream",
    "user_id": "125",
    "session_id": "143",
    "question_id": "143",
    "trace_id": "1746001144320",
    "debug": false,
    "source": "aicubes_agent_77",
    "human_message": "{\"startDate\":\"2023-01-01\",\"chartType\":\"Weekly Candlestick\",\"endDate\":\"2025-04-30\",\"codeName\":\"同花顺\",\"indicator\":[\"MA\",\"MACD\",\"RSI\",\"BOLL\"]}",
    "question": "{\"startDate\":\"2023-01-01\",\"chartType\":\"Weekly Candlestick\",\"endDate\":\"2025-04-30\",\"codeName\":\"MSFT\",\"indicator\":[\"MA\",\"MACD\",\"RSI\",\"BOLL\"]}",
    "stream": false}"""
    req = json.loads(req_json)
    req["human_message"] = query
    req["question"] = query
    resp = request_chain(req)
    if not resp.json()['response']:
        return [f'TickerChart工具，输入{input}，调用失败。']
    resp = resp.json()['response']['result'][0]
    if "media_info" not in resp:
        return [f'TickerChart工具，输入{input}，调用失败。']
    url = resp["media_info"]["url"]

    if url is None:
        return [f'TickerChart工具，输入{input}，调用失败。']
    url_response = requests.get(url)
    image_data = url_response.content
    image = Image.open(BytesIO(image_data))
    filename = url.split("/")[-1]
    # 构建完整的文件路径
    filepath = os.path.join(images_dir, filename)
    # 将图像数据保存到文件
    image = image.convert('RGB')
    image.save(filepath)
    
    return [{"image_path": filepath, "image_url": url}] # 返回的是一个列表，里面是字典

def get_tools_results(tools, images_dir=r'/mnt/HithinkOmniSSD/user_workspace/ganziliang/code/agent/check_images'):
    if images_dir and not os.path.exists(images_dir):
        os.makedirs(images_dir)
    tools_results = []
    prev_tool = None
    
    for tool in tools:
        tool_name = tool['name']
        tool_input = tool['input']
        if prev_tool and tool_name == prev_tool and tool_name == 'Search':
            sleep(1)
        try:
            if tool_name == 'ObtainInfoSummary' or tool_name == 'ObtainInfoContent':
                result = tool_map[tool_name](tool_input, tool['theme_id'])
            elif tool_name == 'TickerChart':
                result = tool_map[tool_name](tool_input, images_dir)
            else:
                result = tool_map[tool_name](tool_input)
            tools_results.extend(result)
        except Exception as e:
            print(f"Error running tool {tool_name}: {e}")
        prev_tool = tool_name
        
    return tools_results # finquery 和 search 返回的是 list，内容是字符串。图片相关工具返回的是 list[list]。
    
    
Below is some example you can use the tools:

res = get_tools_results([                    {'name': 'FinQuery', 'input': '茅台的股票代码'}])
    
The output is 
['取数问句: 茅台的股票代码·\n 取数结果: \n为您找到1条数据\n|股票代码|股票简称|\n|---|---|\n|600519.SH|贵州茅台|\n\n']



res = get_tools_results([                   {'name': 'TickerChart', 'input': '{"codeName": "300584", "chartType": "Daily Candlestick", "startDate": "2025-05-19", "endDate": "2025-06-11", "indicator": ["MA"]}'}])

[{'image_path': '/mnt/HithinkOmniSSD/user_workspace/ganziliang/code/agent/check_images/d990cf52a8fc08e1f72abe5b89f6f6da_750_842.png', 'image_url': 'http://u.thsi.cn/imgsrc/sns/d990cf52a8fc08e1f72abe5b89f6f6da_750_842.png'}]

res = get_tools_results([                    {'name': 'FinQuery', 'input': '茅台的股票代码'}])

['取数问句: 茅台的股票代码·\n 取数结果: \n为您找到1条数据\n|股票代码|股票简称|\n|---|---|\n|600519.SH|贵州茅台|\n\n']

res = get_tools_results([                    {"name":"Search", "input": "马云"},])

['搜索问句: 马云\n标题: 马云(BabySpace创办.....']

The whole work flow is like: The user will give a picture as a input, the model need to first analysis it and use FinQuery Search TickerChart those three tools to find the Stock name, and then return the six serial number of the Stock name.


The first task is to modify the system prompt and create a rl dataset.
For the system prompt, you need to first delete TickerChart and ChartTwinFinder tools description because it doesn't need to use. And you need to write the prompt to teach the model how to use the tool base on the example I give you and latter your design of the tool class. Most importantly, you should guide the model the whole work flow (like first analysis, then make query to ask the finquery tools, after return results, Use the TickerChart to find the Candlestick Patterns, it will return a url picture like:'http://u.thsi.cn/imgsrc/sns/d990cf52a8fc08e1f72abe5b89f6f6da_750_842.png', you may need to download it and compare whether it is similar as the user's input, may be you can also use the search for help, But I don't want the model to strictly follow the pipeline I create, I just want the model to know there is some tool you can use and when you enouncter the issue, you can use those tools)

The second task is to make RL dataset like geo8k, the ground truth is the image's url's first six number after the prefix "Images", ("Images/001209_20251105_20251223.png" -> 001209)

The third task is to build the tool agent, tool class and the yaml, 


First write a system prompt.