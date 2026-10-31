#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# PyTorchとTorchvisionのインポート（AI推論用）
import torch
from torchvision import models

class GoalNavAiNode(Node):
    def __init__(self):
        super().__init__('goal_nav_ai_node')
        
        # 1. パブリッシャーとサブスクライバーの作成
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # 2. 目標地点の設定 (例: x=2.0, y=2.0)
        self.goal_x = 1.0
        self.goal_y = 1.0
        self.distance_threshold = 0.2  # ゴール判定の許容誤差（メートル）
        
        # 現在の状態変数
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_goal_reached = False
        
        # 3. AI推論モデル（MobileNetV2）のロード
        self.get_logger().info("AI推論モデル(CPU)をロード中...")
        self.device = torch.device('cpu')
        # 事前学習済みのMobileNetV2をロード
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info("AI推論モデルのロードが完了しました。")
        
        # 4. 制御ループのタイマー設定 (0.1秒ごと = 10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg: Odometry):
        # 現在の位置情報の取得
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # クォータニオンからYaw角（向き）を計算
        orientation = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def run_ai_inference(self):
        """ゴール到達時に実行するAI推論のダミー/実行関数"""
        self.get_logger().info("★ゴール到達！ AI推論を実行します...")
        
        # ダミーの入力画像テンソルを作成して推論テスト (1, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            output = self.model(dummy_input)
            predicted_class = torch.argmax(output, dim=1).item()
            
        self.get_logger().info(f"★AI推論判定結果 (Class ID): {predicted_class}")

    def control_loop(self):
        if self.is_goal_reached:
            return

        # 目標までの距離と角度の計算
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - self.current_yaw), math.cos(target_yaw - self.current_yaw))

        twist = Twist()

        # ゴール判定
        if distance < self.distance_threshold:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.is_goal_reached = True
            
            # AI推論の実行
            self.run_ai_inference()
            return

        # 自律走行の制御ロジック（比例制御）
        if abs(yaw_error) > 0.2:
            # 向きが大きくずれている場合はその場で旋回
            twist.angular.z = 0.2 if yaw_error > 0 else -0.2
            twist.linear.x = 0.0
        else:
            # 向きが合っていれば前進しながら微調整
            twist.linear.x = min(0.3, 0.5 * distance)
            twist.angular.z = 0.3 * yaw_error

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = GoalNavAiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()