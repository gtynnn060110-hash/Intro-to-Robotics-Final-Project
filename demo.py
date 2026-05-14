# 在mac上跑需要把默认启动命令中的python改成mjpython，不然会报错。
import mujoco
import mujoco.viewer
import time
import os

# 1. 指定机器狗的物理描述文件路径
# 这里使用的是带场景（地板、光照）的完整 A1 模型
xml_path = "mujoco_menagerie/unitree_a1/scene.xml"

# 检查文件是否存在，防止路径配错
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"找不到模型文件，请检查路径是否正确: {xml_path}")

# 2. 核心操作：将 XML 编译为 MuJoCo 物理模型，并初始化数据状态
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("🤖 机器狗加载成功！")
print("👉 隐藏玩法：在弹出的窗口中，用鼠标【双击】狗的身体，然后按住【右键】可以旋转视角，按住【Ctrl + 鼠标左键】可以像上帝之手一样把它拎起来摔在地上！")

# 3. 启动交互式渲染窗口，并让时间开始流动
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 让物理引擎往前计算一小步（默认 2 毫秒）
        mujoco.mj_step(model, data)
        
        # 将最新的物理状态同步到画面上
        viewer.sync()
        
        # 稍微锁一下帧率，防止 CPU 满载跑得太快
        time.sleep(model.opt.timestep)