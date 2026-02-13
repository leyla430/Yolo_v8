import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2

 # deteccion de persona y proximidad

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')
        self.bridge = CvBridge()
        self.current_image = None
        
        # ID 0 = 'person' 
        self.PERSON_CLASS_ID = "0" 

        # Suscripciones
        self.img_sub = self.create_subscription(Image, '/image', self.image_callback, 10)
        self.det_sub = self.create_subscription(Detection2DArray, '/detections_output', self.detection_callback, 10)
        
        self.get_logger().info('--- DETECCIÓN DE PERSONAS INICIADO (ID: 0) ---')

    def image_callback(self, msg):
    
        # visualizacion
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def detection_callback(self, msg):
        if self.current_image is None:
            return
        
        person_count = 0
        
        for det in msg.detections:
            # extract ID
            clase_detectada = str(det.results[0].hypothesis.class_id)
            score = det.results[0].hypothesis.score

            # Solo si el ID es 0 
            if clase_detectada == self.PERSON_CLASS_ID:
                person_count += 1
                
                # lógica de proximidad
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                
                self.get_logger().info(
                    f'PERSONA DETECTADA ({person_count}) - Posición: x={cx:.1f}, y={cy:.1f}'
                )

        if person_count == 0:
            pass

def main():
    rclpy.init()
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
