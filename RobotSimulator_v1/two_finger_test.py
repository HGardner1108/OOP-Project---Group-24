import pybullet as p
import time
#p.connect(p.UDP,"192.168.86.100")
import pybullet_data

cid = p.connect(p.SHARED_MEMORY)

if (cid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
#disable rendering during loading makes it much faster
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
objects = [
    p.loadURDF("pr2_gripper.urdf", 0.500000, 0.300006, 0.700000, -0.000000, -0.000000, -0.000031,
               1.000000)
]
pr2_gripper = objects[0]

jointPositions = [0.550569, 0.000000, 0.549657, 0.000000]
for jointIndex in range(p.getNumJoints(pr2_gripper)):
  p.resetJointState(pr2_gripper, jointIndex, jointPositions[jointIndex])

pr2_cid = p.createConstraint(pr2_gripper, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0],
                             [0.500000, 0.300006, 0.700000])
print("pr2_cid")
print(pr2_cid)


#p.changeConstraint(pr2_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.setGravity(0.000000, 0.000000, 0.000000)
p.setGravity(0, 0, -10)

p.setRealTimeSimulation(1)

done = False
while (1):
    p.setGravity(0, 0, -10)
    input("enter to close the gripper")
    p.setJointMotorControl2(pr2_gripper, 0, p.POSITION_CONTROL, 
                                targetPosition=0.1, maxVelocity=1,force=1)
    p.setJointMotorControl2(pr2_gripper, 2, p.POSITION_CONTROL, 
                                targetPosition=0.1, maxVelocity=1,force=1)
    input("enter to open the gripper")
    p.setJointMotorControl2(pr2_gripper, 0, p.POSITION_CONTROL, 
                                targetPosition=0.0, maxVelocity=1,force=1)
    p.setJointMotorControl2(pr2_gripper, 2, p.POSITION_CONTROL, 
                                targetPosition=0.0, maxVelocity=1,force=1)
p.disconnect()