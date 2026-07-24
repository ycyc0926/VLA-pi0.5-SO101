# from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
# import time

# # 尝试不带 ID 运行，或带上你之前的 ID
# cfg = SO101FollowerConfig(port="/dev/ttyACM2", id="my_awesome_follower_arm")
# robot = SO101Follower(cfg)
# robot.connect()

# for _ in range(5):
#     obs = robot.get_observation()
#     print(f"Raw Observation: {obs}")
#     time.sleep(0.5)

# robot.disconnect()


import cv2
import time

def test_camera_fps(index, duration=10):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {index}")
        return

    print(f"正在测试摄像头 {index}，持续 {duration} 秒...")
    
    # 尝试设置与推理代码一致的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    num_frames = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("读取失败")
            break
        num_frames += 1

    end_time = time.time()
    seconds = end_time - start_time
    fps = num_frames / seconds

    print(f"测试结束！")
    print(f"总帧数: {num_frames}")
    print(f"总时间: {seconds:.2f} 秒")
    print(f"实际平均 FPS: {fps:.2f}")

    cap.release()

# 针对你代码中使用的摄像头 index 进行测试
test_camera_fps(4)  # 测试 top cam
test_camera_fps(2)  # 测试 wrist cam