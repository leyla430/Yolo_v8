import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

# resize camara RGB para ser compatible con Yolo yevitar el desface del bounding box 

class RQTFixer(Node):
    def __init__(self):
        super().__init__('rqt_fixer_node')
        self.subscription = self.create_subscription(
            Image, '/camera/color_image', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/image', 10)
        self.get_logger().info('Fixer con Padding: 640x480 -> 640x640')

    def listener_callback(self, data):
        if not data.data or len(data.data) < 921600:
            return
        try:
            img_data = np.frombuffer(data.data, dtype=np.uint8)
            img = img_data.reshape((480, 640, 3))
            
            # cuadrado de 640x640
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # page image
            canvas[80:560, 0:640] = img
            
            msg = Image()
            msg.header = data.header # timestamp 
            msg.height = 640
            msg.width = 640
            msg.encoding = 'bgr8'
            msg.step = 640 * 3
            msg.data = canvas.tobytes()
            
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = RQTFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
