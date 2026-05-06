/data2/wcl/MemGUI-Bench/results/session-memgui-v26050510-new-owl-all
输入上述路径，读取上述路径的所有任务的evaluation_summary.json文件
/data2/wcl/MemGUI-Bench/results/session-memgui-v26050510-new-owl-all/128-CompareCarsAndResearch/Qwen3VL/attempt_1/evaluation_summary.json
关注下面的字段，检查是否有的"final_reason"，比如∵找不到app，任务无法完成，"final_result"为1（成功），app中没有内容（没网路）等可能需要重点关注的情况
使用gemini-3-pro-preview-new-priority模型进行判断，辅助人类找到需要检查关键的任务，可以多线程处理
    "task_identifier": "128-CompareCarsAndResearch",
    "task_description": "Open Cars.co.za app, search for a 'Ford Ranger', and remember the price of the first result. Then search for a 'Toyota Hilux' and remember the price of the first result. Open the Calculator app and find the price difference. Open the joplin app to note which car is more expensive and by how much. Finally, open the bing app and search for \"Ford Ranger vs Toyota Hilux reliability\". Stay on the search results page.",
    "final_result": 0,
    "final_reason": "The agent was unable to complete any of the core requirements of the task due to a persistent network error. It could not retrieve the prices for the cars, and therefore could not perform the calculation, note the result, or proceed to the final search. The agent made multiple attempts to circumvent the issue by restarting the app and using a web browser, but the lack of an internet connection made the task impossible to complete. The final state does not meet any of the task's objectives.",
    