Multi-Agent Modeling SeeUPO abstracts multi-turn interaction tasks into sequentially-decision multiagent single-turn bandit problems, where each turn is mapped to a virtual agent t ∈ {1, 2, . . . , T} (as depicted in Figure 2). All agents share a common global state s0 ∈ SS (the initial task state). The sequencelevel action at ∈ AS of agent t corresponds to the complete response of the t-th turn. The joint action a1:T = (a1, a2, . . . , aT ) denotes the concatenation of all agents’ actions. Each agent t’s sequence-level policy πt(at|s0, a1:t−1) takes as input the global state s0 and the action history from preceding agents a1:t−1. The state transition function is implicitly modeled through sequentially executed policies: the evolution of the interaction is determined by agent t’s action selected according to policy πt(·|s0, a1:t−1). We adopt a shared reward (Team-Reward) mechanism r(s0, a1:T ), where the team reward equals the final task reward or cumulative return across all turns, ensuring all agents jointly optimize the global objective. We emphasize that this multi-agent modeling only exhibits a training-phase specificity and offers no inherent advantages for execution optimization. This abstraction constitutes a methodological shift in data treatment during the training phase, but does not correspond to a functional mechanism for multi-agent coordination during actual execution. Policy Update After modeling is completed, the optimization of Sequence-level policies is transformed into optimizing the joint policy of a multi-agent system.† SeeUPO adopts the sequential update mechanism from the HAML framework (see Appendix A), with a crucial design choice: the update order is set to the reverse of the execution order (T → T−1 → · · · → 1). This reverse update order not only resolves agent-level update conflicts by updating policies turn by turn, but also enables backward induction to achieve global optimality (see Theorem 2 in Appendix B). At each iteration k, the algorithm updates each agent’s policy sequentially following the reverse order of execution (T → T−1 → · · · → 1). The policy update process consists of three key components: (1) Policy Update Rule. For turn t in the reverse update order, by substituting the HAML into the update rule, the policy update can be expressed as: ˆπt k+1 = arg max ¯πt ∈U tˆπk ( ˆπt k ) Es0∼β ˆπk " Eat+1:T ∼ ˆπt+1:T k+1 ,at ∼ ¯πt h Atˆπk (s0, at, at+1:T ) i − Dtˆπk ( ¯πt | s0, ˆπt+1:T k+1 ) # , (3) where s0 is the sequence-level joint state (i.e., the global initial state), ˆπk denotes the joint policy at iteration k, U tˆπk ( ˆπt k) is the neighborhood operator for turn t, β ˆπk is the sampling state distribution, Eat+1:T ∼ ˆπt+1:T k+1 ,at ∼ ¯πt h Atˆπk i is the expectation of the local advantage function for turn t, and Dtˆπk is the drift functional. Crucially, under the reverse update order, each turn t considers the already-updated policies ˆπt+1:T k+1 of subsequent turns, ensuring coordination in the update process and enabling backward induction. (2) Local Advantage Function Computation. To compute the expectation of the local advantage function Eat+1:T ∼ ˆπt+1:T k+1 ,at ∼ ¯πt h Atˆπk i , SeeUPO leverages the global advantage function. Given the global advantage function ˆA ˆπk (s0, a1:T ), this expectation can be estimated: Eat+1:T ∼ ˆπt+1:T k+1 ,at ∼ ¯πt h Atˆπk (s0, at, at+1:T ) i = Ea1:T ∼ ˆπk " ¯πt(at|s0, a1:t−1) ˆπt k(at|s0, a1:t−1) − 1 ! · ˆπt+1:T k+1 (at+1:T |s0, a1:t) ˆπt+1:T k (at+1:T |s0, a1:t) · ˆA ˆπk (s0, a1:T ) # , (4) where the first term  ¯πt (at |s0,a1:t−1) ˆπt k (at |s0,a1:t−1) − 1  involves the candidate policy ¯πt (to be optimized) and the current policy ˆπt k, while the second ratio ˆπt+1:T k+1 (at+1:T |s0,a1:t ) ˆπt+1:T k (at+1:T |s0,a1:t ) involves the already-updated joint policy of subsequent turns ˆπt+1:T k+1 and the previous policy ˆπt+1:T k . Note that the −1 term in the first factor has zero gradient with respect to ¯πt and can be omitted in practical gradient computation. The computation 7  of the local advantage function effectively performs implicit credit assignment across turns (Zhong et al., 2024), as it decomposes the global advantage into turn-specific contributions by incorporating the importance sampling ratios from subsequently updated turns. (3) Global Advantage Function Computation. In the bandit setting, the global advantage function ˆA ˆπk (s0, a1:T ) can be estimated directly from sampled rewards. Specifically, for a given initial state s0 and sequence-level joint action a1:T, the global advantage function degenerates to: ˆA ˆπk (s0, a1:T ) = r(s0, a1:T ) − Ea′1:T ∼ ˆπk (·|s0)[r(s0, a′1:T )], (5) where r(s0, a1:T ) is the immediate reward and Ea′1:T ∼ ˆπk (·|s0)[r(s0, a′1:T )] is the expected reward under the current policy. This formulation provides an unbiased estimate of the advantage function in the bandit setting. Theoretical Guarantees SeeUPO inherits the monotonic improvement guarantee from the HAML framework (Theorem 1 in Appendix A). Beyond this, we establish a stronger result for the multi-turn contextual bandit setting: the reverse update order guarantees convergence to the globally optimal policy (Theorem 2 in Appendix B). The key insight is that, unlike general cooperative games where random update orders are required for Nash equilibrium convergence, the fixed sequential execution order in our setting enable backward induction. Specifically, the reverse update order ensures that when updating turn t, all subsequent turns t + 1, . . . , T have already been updated to their optimal policies given their continuation values. This allows each turn to optimize against the true optimal continuation value V∗, yielding global optimality. The complete proof is provided in Appendix B. 4.2 Practical Methods Algorithm 1: SeeUPPO-GRAE Input: Initial sequence-level joint policy π0 with parameters θ0, maximum turns T, batch size B, group size G, clipping parameter ϵ, learning rate α Output: Optimized policy πK after K iterations 1. Initialize π0 with parameters θ0 2. For k = 0, 1, . . . , K − 1: (a) Data Collection: • Sample dataset Dk = {(s0, a1:T, r)} by: – For each of B initial states s0, sample G trajectories a1:T and collect rewards r(s0, a1:T ) • Organize data into T sample pools: for each turn t ∈ {1, . . . , T}, construct pool Dt = {(s0, a1:t−1, at)} (b) Joint Advantage Estimation: • For each (s0, a1:T ) ∈ DT : – Compute joint advantage: ˆA ˆπk (s0, a1:T ) = r(s0, a1:T ) − ¯r(s0) – Initialize MT+1(s0, a1:T ) = ˆA ˆπk (s0, a1:T ) (c) Sequential Policy Update (Reverse Order): • For t = T, T−1, . . . , 1: – Update policy parameters to obtain πt θk+1 via Equation. 6: – If t > 1: * Update Mt for next batch via Equation. 8 – Else: * Set θk+1 = θ1 k+1 3. Return πK In this part, we present a concrete example of SeeUPO (Algorithm 1 presents the pseudocode). The practical implementation instantiates the theoretical framework in two key aspects: (1) adopting a PPOstyle clipping mechanism to implement the mirror operator through gradient-based updates, and (2) using GRAE for joint advantage estimation. This approach essentially combines GRAE with HAPPO (Zhong et al., 2024). We refer to this practical algorithm as SeeUPPO-GRAE, though for convenience, we still refer to it as SeeUPO in the remainder of this paper. We emphasize that SeeUPPO-GRAE is not the only instantiation of SeeUPO—the theoretical framework admits various variants by substituting different components, such as replacing the PPO-style clipping with TRPO-style trust region constraints, or replacing GRAE with other advantage estimators. The algorithm operates iteratively, with each iteration comprising data collection, advantage estimation, and sequential policy updates. SeeUPO adopts a turn-oriented batch construction approach that separately organizes samples from identical turns, as illustrated in Figure 3. This approach enables sequential policy updates by maintaining turn-level sample pools, in contrast to methods that construct batches using entire trajectories or concatenated sliced turns. For tasks with fewer than the maximum T turns, placeholder samples (e.g., Sample 6 in the figure) are introduced as no-op (null action) samples. Note that the figure demonstrates batch construction patterns using the React + Reasoning-Augmented Template paradigm (Zhai et al., 2025). PPO-Style Policy Update In our multi-turn RL setting, all turns share the same policy parameters θ. We use πθt k+1 to denote the policy after updating turn t’s data in iteration k. For notational clarity, we denote (s0, a1:T ) as a joint trajectory sample, where s0 is the initial state (query) and a1:T is the sequence-level joint action as defined in Section 4.1. Specifically, for turn t in the reverse update order (T → T−1 → · · · → 1) at iteration k, the policy update is performed to obtain πθt k+1 by computing the gradient of the policy parameters θ with respect to the following expectation: ∇θE(s0,a1:t−1,at )∼Dt h min  rt(θ)Mt+1(s0, a1:T ), clip(rt(θ), 1 ± ϵ)Mt+1(s0, a1:T ) i , (6) where Dt = {(s0, a1:t−1, at)} is the turn-specific sample pool constructed during data collection (see Algorithm 1), containing samples organized by turn t. The sequence-level importance sampling ratio rt(θ) = πθ(at |s0,a1:t−1) πθk (at |s0,a1:t−1) for turn t is computed in a manner similar to GSPO, where at denotes the action at turn t and (s0, a1:t−1) is the conditioning context (initial state and previous actions). ϵ is the clipping parameter, and Mt+1(s0, a1:T ) is a maintained quantity that captures the sequential advantage information from subsequently updated turns. The quantity Mt(s0, a1:T ) is initialized and updated sequentially to incorporate the importance sampling ratios from previously updated turns. Specifically, we initialize: MT+1(s0, a1:T ) = ˆA ˆπk (s0, a1:T ), (7) where ˆA ˆπk (s0, a1:T ) is the global advantage estimate (computed via GRAE as described below). After updating turn t, Mt(s0, a1:T ) is computed recursively: Mt(s0, a1:T ) = πθt k+1 (at|s0, a1:t−1) πθk (at|s0, a1:t−1) · Mt+1(s0, a1:T ), (8) where θt k+1 denotes the parameters after optimizing turn t, and θk denotes the parameters at the beginning of iteration k (i.e., the reference policy used for sampling). Due to parameter sharing, this sequential update mechanism ensures that Mt(s0, a1:T ) incorporates the importance sampling ratios from all previously updated turns, matching the expectation structure in Equation 4. GRAE-based Advantage Estimation In the bandit setting, the global advantage function ˆA ˆπk (s0, a1:T ) can be estimated directly from sampled rewards, as established in Equation 5. For each initial state in the batch, SeeUPO samples G different joint actions and collects the corresponding Team-Rewards. The global advantage estimate is computed as: ˆA ˆπk (s0, a1:T ) = r(s0, a1:T ) − ¯r(s0), (9) where ¯r(s0) is the mean reward over G trajectories sampled from the same initial state s0, serving as a Monte Carlo estimator of V ˆπk (s0) = Ea1:T ∼ ˆπk (·|s0)[r(s0, a1:T )]. This approach provides an unbiased estimate of the advantage function in the bandit setting, without requiring a separate critic (see Appendix H for detailed analysis). In practice, we apply batch-level normalization to the advantage estimates for numerical stability. Specifically, we normalize all advantage estimates in a batch as: ˜A = ( ˆA − μB )/σB , where μB is the batch mean and σB is the batch standard deviation. This normalization approach maintains theoretical convergence guarantees while improving training stability: since μB and σB are constants independent of the candidate policy, the argmax of the optimization problem remains unchanged, leaving the drift functional completely unaffected (see Appendix H.3.3 for detailed analysis). This is in contrast to group normalization which applies state-dependent scaling factors that can violate these properties (see Appendix H.3). Moreover, experimental results in Section 5.3.2 demonstrate that batch-level normalization performs comparably to group normalization and no normalization, while preserving the theoretical convergence properties.

Above is a algorithm called seeUPO, which is designed for multi-turn RL: which is perfectly for my scenior. My task will using function call:

The whole generation code is below

class StockChartAgentLoop(AgentLoopBase):
    """
    Multi-turn RL agent loop for stock chart identification.

    Differences from the generic ToolAgentLoop:
      * Adds a dedicated SCORING state that fires calc_stock_reward when
        the model outputs <FINISHED>.
      * Injects tool-call results that may include images (TickerChart)
        back into the conversation as multi-modal content.
      * Enforces a hard cap on external tool calls (max_tool_calls) to
        prevent runaway API usage during rollout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mt = self.rollout_config.multi_turn
        self.max_user_turns          = mt.max_user_turns
        self.max_assistant_turns     = mt.max_assistant_turns
        self.max_parallel_calls      = mt.max_parallel_calls
        self.max_tool_response_length = mt.max_tool_response_length
        self.tool_response_truncate_side = mt.tool_response_truncate_side
        self.prompt_length           = self.rollout_config.prompt_length
        self.response_length         = self.rollout_config.response_length

        # External tools: FinQuery, Search, TickerChart
        tool_config_path = mt.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools       = {t.name: t for t in tool_list}
        self.tool_schemas = [
            t.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for t in tool_list
        ]

        self.tool_parser      = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.tool_parser_name = mt.format
        


    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        request_id   = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics={},
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # ── initialise the reward tool instance ──────────────────────────
        reward_tool   = self.tools.get("calc_stock_reward")
        reward_instance_id = None
        if reward_tool is not None:
            rw_kwargs = tools_kwargs.get("calc_stock_reward", {})
            reward_instance_id, _ = await reward_tool.create(
                **rw_kwargs.get("create_kwargs", {})
            )

        # ── state machine ────────────────────────────────────────────────
        state = StockAgentState.PENDING
        while state != StockAgentState.TERMINATED:
            if state == StockAgentState.PENDING:
                state = await self._handle_pending(agent_data, sampling_params)
            elif state == StockAgentState.GENERATING:
                state = await self._handle_generating(
                    agent_data, sampling_params, reward_instance_id
                )
            elif state == StockAgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools(agent_data)
                
            else:
                logger.error(f"Unknown state {state}, terminating.")
                state = StockAgentState.TERMINATED

        # ── release reward tool ──────────────────────────────────────────
        if reward_tool is not None and reward_instance_id is not None:
            await reward_tool.release(reward_instance_id)

        # ── build output ─────────────────────────────────────────────────
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids   = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        mm_data = {}
        if agent_data.image_data is not None:
            mm_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            mm_data["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=mm_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
        })
        return output

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_pending(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> StockAgentState:
        """Tokenise the initial prompt (system + user message with image)."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return StockAgentState.GENERATING

    async def _handle_generating(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        reward_instance_id: str | None,
    ) -> StockAgentState:
        """Run the LLM for one turn and decide the next state."""
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # ── book-keeping ─────────────────────────────────────────────────
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        else:
            agent_data.metrics["num_preempted"] += (
                output.num_preempted if output.num_preempted is not None else 0
            )

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_gs = output.extra_fields.get("max_global_steps")
            if max_gs:
                agent_data.extra_fields["max_global_steps"] = max_gs

        agent_data.assistant_turns += 1
        agent_data.response_ids     = output.token_ids
        agent_data.prompt_ids      += agent_data.response_ids
        agent_data.response_mask   += [1] * len(agent_data.response_ids)

        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # ── hard termination guards ───────────────────────────────────────
        if len(agent_data.response_mask) >= self.response_length:
            return StockAgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return StockAgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return StockAgentState.TERMINATED

        # ── decode response to inspect content ───────────────────────────
        response_text: str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=False),
        )

        # ── check for <FINISHED> ──────────────────────────────────────────
        if "<FINISHED>" in response_text:
            # Record assistant message before scoring
            agent_data.messages.append({"role": "assistant", "content": response_text})
            return StockAgentState.SCORING

        # ── check for tool calls ──────────────────────────────────────────
        tools = [t.tool_schema for t in self.tools.values()
                 if t.name != "calc_stock_reward"]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            output.token_ids, tools
        )

        if agent_data.tool_calls:
            agent_data.messages.append({"role": "assistant", "content": response_text})
            return StockAgentState.PROCESSING_TOOLS

        # No tool calls and no <FINISHED>: the model is mid-thought; keep generating
        # (This handles the case where the model fills the context with reasoning text.)
        agent_data.messages.append({"role": "assistant", "content": response_text})
        return StockAgentState.TERMINATED

    async def _handle_processing_tools(self, agent_data: AgentData) -> StockAgentState:
        """
        Execute FinQuery / Search / TickerChart tool calls in parallel,
        inject results (including images) back into the conversation.
        """
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # Fire external tool calls (skip calc_stock_reward here — that has its own state)
        tasks = []
        tool_call_names = []
        for tc in agent_data.tool_calls[: self.max_parallel_calls]:
            if tc.name == "calc_stock_reward":
                # Should not appear here, but guard just in case
                continue
            tasks.append(self._call_external_tool(tc, agent_data))
            tool_call_names.append(tc.name)

        if not tasks:
            # Nothing to execute (all calls were filtered out)
            return StockAgentState.GENERATING

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
            
    #     self._log_turn_debug(
    #     agent_data,
    #     turn_type="TOOL_RESPONSE",
    #     prompt_ids=agent_data.prompt_ids,  # full prompt so far
    #     tool_responses=[r[0].text for r in responses],  # ToolResponse.text per call
    # )

        for tool_response, tool_reward in responses:
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            # Build the tool-result message
            if tool_response.image:
                content = []
                content.append({"type": "image"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
                # Collect new images for multi-modal prompt update
                imgs = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                new_images_this_turn.extend(i for i in imgs if i is not None)
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

        agent_data.messages.extend(add_messages)

        # Tokenise the tool-result messages
        images  = new_images_this_turn if new_images_this_turn else None
        response_ids = await self.apply_chat_template(
            add_messages,
            images=images,
            videos=None,
            remove_system_prompt=True,
        )

        # Respect response length budget
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return StockAgentState.TERMINATED

        # Update image data
        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids    += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        agent_data.user_turns += 1
        return StockAgentState.GENERATING


    # ------------------------------------------------------------------
    # External tool dispatch  (FinQuery / Search / TickerChart)
    # ------------------------------------------------------------------

    async def _call_external_tool(
        self,
        tool_call: FunctionCall,
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float | None]:
        """
        Dispatch a single FinQuery / Search / TickerChart call.

        Returns (ToolResponse, optional_reward).
        """
        tool_name = tool_call.name
        tool = self.tools.get(tool_name)
        if tool is None:
            return (
                ToolResponse(text=f"未知工具：{tool_name}。可用工具：{list(self.tools.keys())}"),
                None,
            )

        instance_id = None
        try:
            tool_args = json.loads(tool_call.arguments)
            kwargs    = agent_data.tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(
                create_kwargs=kwargs.get("create_kwargs", {})
            )
            tool_response, tool_reward, _ = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"External tool '{tool_name}' error: {e}")
            return ToolResponse(text=f"工具调用失败 ({tool_name}): {e}"), None
        finally:
            if tool is not None and instance_id is not None:
                await tool.release(instance_id)

        # Truncate over-long responses
        text = tool_response.text or ""
        if len(text) > self.max_tool_response_length:
            side = self.tool_response_truncate_side
            L    = self.max_tool_response_length
            if side == "left":
                text = text[:L] + "...(truncated)"
            elif side == "right":
                text = "(truncated)..." + text[-L:]
            else:
                half = L // 2
                text = text[:half] + "...(truncated)..." + text[-half:]

        # Rebuild response with possibly-truncated text but preserve image/video
        kw = {"text": text}
        for attr in ("image", "video"):
            val = getattr(tool_response, attr, None)
            if val is not None:
                kw[attr] = val

        return ToolResponse(**kw), tool_reward
    

So the whole procedure is 

     while state != StockAgentState.TERMINATED:
            if state == StockAgentState.PENDING:
                state = await self._handle_pending(agent_data, sampling_params)
            elif state == StockAgentState.GENERATING:
                state = await self._handle_generating(
                    agent_data, sampling_params, reward_instance_id
                )
            elif state == StockAgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools(agent_data)
                
            else:
                logger.error(f"Unknown state {state}, terminating.")
                state = StockAgentState.TERMINATED
                
                
And for the extract_tool_calls function


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
       
Refer to this paper: I want to split the trajectory BY THE TOOl CALL。 But when compute the loss: for example in the GRPO:

@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics

It will directly the reward signal into the whole trajectory, so you need to design some method to split the whole trajectory into several segment and compute the loss individually

Below is the whole loop of update policy. I guess you also need to modify:

Below is the back bone of the RL training:
    

  for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                            

I suppose you need to modify the 
_compute_old_log_prob and update_policy

Below is _compute_old_log_prob

 def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        prev_modes = [m.training for m in self.actor_module]
        for module in self.actor_module:
            module.eval()
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]

            if self.enable_routing_replay and self.config.router_replay.mode == "R3":
                assert "routed_experts" in data.batch.keys(), "routed_experts must be in data.batch.keys()"
                select_keys.append("routed_experts")

            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(
                    data,
                    forward_only=True,
                    post_process_fn=compute_logprobs_fn,
                    calculate_entropy=calculate_entropy,
                    use_dynamic_bsz=use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    if calculate_entropy:
                        log_probs = [o[0]["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    else:
                        log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        log_probs = log_probs[revert_indices]
                else:
                    log_probs = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )
                log_probs = log_probs.to(get_device_id())
                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                log_probs = log_probs.to("cpu")
                if calculate_entropy:
                    # Note that o[0] is metrics, o[1] is entropy
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                        entropys = entropys.to(torch.float32)
                        if use_dynamic_bsz:
                            indices = output["indices"]
                            indices = list(itertools.chain.from_iterable(indices))
                            assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                            entropys = entropys[revert_indices]
                    else:
                        entropys = torch.empty(
                            size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                        )
                    # broadcast across pp ranks
                    entropys = entropys.to(get_device_id())
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )
                    entropys = entropys.to("cpu")
                layers_topk_idx = None

                if RouterReplayHelper.is_r2_record_action(self.tf_config):
                    # (bs, max_seq_len/response_len,local_layer_num,topk)
                    layers_topk_idx = output["mini_layer_topk_idx_tensor"].to(torch.uint8)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == layers_topk_idx.size(0), f"{len(indices)} vs. {layers_topk_idx.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        layers_topk_idx = layers_topk_idx[revert_indices]
                    layers_topk_idx = pp_gather(layers_topk_idx, self.tf_config)
        # add empty cache after each compute
        get_torch_device().empty_cache()

        for module, mode in zip(self.actor_module, prev_modes, strict=False):
            module.train(mode)
        return log_probs, entropys, layers_topk_idx
    
Below is the update_policy


 @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto], enable_mtp: bool = False) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

            enable_mtp (bool, optional): whether to enable MTP communication

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        for data in dataloader:
            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )

            mtp_losses = metric_micro_batch.get("mtp_losses", None)
            if mtp_losses is not None:
                # mtp_losses is now in format: [{"mtp_losses/mtp_1_loss": [value1], "mtp_losses/mtp_2_loss": [value2]}]
                for mtp_metrics_dict in mtp_losses:
                    append_to_dict(metrics, mtp_metrics_dict)

            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()

        self.actor_optimizer.zero_grad()
        get_torch_device().empty_cache()
        return metrics

I don't use KL and critic, So you can ignore critic update policy and ref log prob
