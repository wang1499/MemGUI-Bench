将qwen agent的提示词部分重构，支持qwen v1v2 都从/data2/wcl/MemGUI-Bench/framework/models/AndroidWorld/android_world/agents/qwen_propmt.py中统一加载提示词方便管理
就像mobile_use_tool, write_memories_tool, todo_rules_description 这些提示词，都从qwen_propmt.py中加载，


给包含answer的部分任务在评估时显示添加答案作为评估参考
如当任务是042-CheckSsdCompatibilityAndPrice时，答案应该为不兼容，以及对应的详细理由。（如果你不知道理由可以空出来，让人类后续自己来扩展，比如读取文件，判断任务id是否存在参考答案，有就加上）

“”"The agent successfully completed all steps of the task. It correctly searched for the specified SSD on Amazon and gathered its speed information (1,050 MB/s as per the description of step 6). It then searched for the PS5's speed requirement on Bing and correctly identified it as '5 GB/s (at least)' from the search results shown in the final screenshots. Finally, it correctly compared the two speeds (1,050 MB/s is less than 5,000 MB/s) and provided the correct final answer, 'Not compatible', as instructed.",
“”

补充判断的提示词，现在有这样的情况如下2个判断给出了不同的结果，判断二认为VLM存在幻觉，你需要补充，VLM是辅助批判器，并不是任务执行器的一部分。（大部分时候可以相信辅助vlm）  
任务041-CheckRamCompatibilityAndPrice答案应该是兼容，参考价格是$429.99（模型输出可能不同，价格接近与模型在商场查询相同即可）
"The agent failed to provide the correct final answer. According to the VLM description for step 11, the first RAM module had a clock speed of 6000MHz and a price of $429.99. The final screenshots show the motherboard supports speeds up to 7800 MT/s. Since 6000MHz is less than 7800 MT/s, the RAM is compatible, and the correct answer should have been the price, '$429.99'. However, in step 23 the agent first incorrectly answered 'Not compatible', and then in subsequent steps answered with an incorrect price of '188.99'.",
"The agent successfully completed all sub-tasks. It correctly navigated to the Amazon app, searched for the specified RAM, and applied the 'Corsair' brand filter. The VLM description for step 9 notes the clock speed as 6000MHz. The agent then correctly switched to the Bing app and found that the motherboard supports a maximum speed of 7800MHz, as shown in the final screenshots. Since 6000MHz is less than 7800MHz, the RAM is compatible, and the correct final action is to provide the price. The agent answered with a price of '$119.99'. While the VLM description for step 9 mentions a price of '$429.99', this is an extremely unlikely price for the specified RAM, whereas '$119.99' is a highly plausible price. Given that the VLM descriptions can be incomplete or erroneous, and the agent correctly performed all other logical steps of the task, it is reasonable to conclude that the agent correctly identified the price as '$119.99' and the VLM description was flawed. Therefore, the task is considered a success.",
