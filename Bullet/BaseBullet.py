'''
    子弹的基础类
'''
import pygame
import json
import os
import math
from Parameter import BULLET_SPEED, BULLET_DAMAGE

class BaseBullet:
    def __init__(self, projectile_id, bullet_image_path, shooter_id=0, position=(0.0, 0.0), velocity_direction=(1.0, 0.0), size=(0.0, 0.0), bounding_box=None, lifetime=3.0, speed_rate=1.0, damage_rate=1.0, penetration=[1.0, 1.0, 1.0] , is_explosive=False, explosion_radius=0.0, explosion_image_path=None):
        # 基本信息
        self.id = projectile_id
        self.shooter_id = shooter_id
        self.image_path = bullet_image_path
        self.image = self.load_image(bullet_image_path) if bullet_image_path else None  
        self.size = size if not size==(0.0, 0.0) else self.image.get_size()
        
        # 攻击距离，速度，伤害等属性
        self.max_lifetime = lifetime
        self.speed_rate = speed_rate        # speed = speed_rate * BULLET_SPEED
        self.damage_rate = damage_rate      # damage = damage_rate * BULLET_DAMAGE
        self.penetration = penetration      # 对轻，对中，对重伤害比例
        self.speed = BULLET_SPEED * self.speed_rate     # 速度的值
        self.damage = BULLET_DAMAGE * self.damage_rate
        
        # 是否会爆炸以及爆炸效果
        self.is_explosive = is_explosive
        self.explosion_radius = explosion_radius
        self.explosion_image_path = explosion_image_path
        self.explosion_image = pygame.image.load(explosion_image_path) if explosion_image_path else None
        
        # 实时属性
        self.lifetime = lifetime                # 子弹的剩余寿命
        self.position = position
        self.velocity_direction = velocity_direction
        self.bounding_box = bounding_box if bounding_box else self._update_bounding_box      # 碰撞箱，pygame.Rect对象
        self.velocity = self.cal_velocity()     # 速度向量

        # 预留属性
        self.is_active = True
        self.has_collided = False
    
    def cal_velocity(self):
        '''
        根据speed和velocity_direction计算速度向量
        '''
        if self.velocity_direction[0]==0 and self.velocity_direction[1]==0:
            direction = (1.0, 0.0)
        else:
            direction = self.velocity_direction
        
        dir_length = math.sqrt(direction[0]**2 + direction[1]**2)
        vx = direction[0] / dir_length * self.speed
        vy = direction[1] / dir_length * self.speed
        return (vx, vy)
    
    def load_image(self, image_path):
        try:
            return pygame.image.load(image_path)
        except:
            print(f"Warning: Cannot load body image: {image_path}")
            return None
        
    def to_json(self, file_path="Bullet/json/bullet_default.json"):
        """
        将子弹的基本属性保存为JSON文件
        使用示例：
            bullet = BaseBullet()
            bullet.to_json("Bullet/json/bullet_default.json")
        """
        # 构建序列化数据字典
        data = {
            "id": self.id,
            "shooter_id": self.shooter_id,
            "image_path": self.image_path,
            
            "position": {
                "x": self.position[0] if hasattr(self.position, '__len__') else 0.0,
                "y": self.position[1] if hasattr(self.position, '__len__') and len(self.position) > 1 else 0.0
            },
            "velocity_direction": {
                "x": self.velocity_direction[0] if hasattr(self.velocity_direction, '__len__') else 0.0,
                "y": self.velocity_direction[1] if hasattr(self.velocity_direction, '__len__') and len(self.velocity_direction) > 1 else 0.0
            },
            "size": {
                "width": self.size[0] if hasattr(self.size, '__len__') else 0.0,
                "height": self.size[1] if hasattr(self.size, '__len__') and len(self.size) > 1 else 0.0
            },
            
            "bounding_box": {
                "x": self.bounding_box.x if self.bounding_box else 0,
                "y": self.bounding_box.y if self.bounding_box else 0,
                "width": self.bounding_box.width if self.bounding_box else 0,
                "height": self.bounding_box.height if self.bounding_box else 0
            } if self.bounding_box else None,
            
            "lifetime": self.lifetime,
            "speed_rate": self.speed_rate,
            "damage_rate": self.damage_rate,
            "penetration": self.penetration,
            
            "is_explosive": self.is_explosive,
            "explosion_radius": self.explosion_radius,
            "explosion_image_path": self.explosion_image_path,
            
            "is_active": self.is_active,
            "has_collided": self.has_collided
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    
    @classmethod
    def from_json(cls, file_path="Bullet/json/bullet_default.json"):
        """
        从JSON文件加载子弹属性并创建新的子弹实例
        使用示例：
            loaded_bullet = BaseBullet.from_json("Bullet/json/bullet_default.json")
            if loaded_bullet:
                print(f"加载的子弹ID: {loaded_bullet.id}")
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从数据中提取属性
            projectile_id = data.get("id", "")
            shooter_id = data.get("shooter_id", "")
            bullet_image_path = data.get("image_path")
            
            position_data = data.get("position", {})
            position = (position_data.get("x", 0.0), position_data.get("y", 0.0))
            
            velocity_direction_data = data.get("velocity_direction", {})
            velocity_direction = (velocity_direction_data.get("x", 0.0), velocity_direction_data.get("y", 0.0))
            
            size_data = data.get("size", {})
            size = (size_data.get("width", 0.0), size_data.get("height", 0.0))
            
            bounding_box_data = data.get("bounding_box")
            bounding_box = None
            if bounding_box_data:
                bounding_box = pygame.Rect(
                    bounding_box_data.get("x", 0),
                    bounding_box_data.get("y", 0),
                    bounding_box_data.get("width", 0),
                    bounding_box_data.get("height", 0)
                )
            
            lifetime = data.get("lifetime", 3.0)
            speed_rate = data.get("speed_rate", 1.0)
            damage_rate = data.get("damage_rate", 1.0)
            penetration = data.get("penetration", [1.0, 1.0, 1.0])
            
            is_explosive = data.get("is_explosive", False)
            explosion_radius = data.get("explosion_radius", 0.0)
            explosion_image_path = data.get("explosion_image_path")
            
            bullet = cls(
                projectile_id=projectile_id,
                shooter_id=shooter_id,
                bullet_image_path=bullet_image_path,
                position=position,
                velocity_direction=velocity_direction,
                size=size,
                bounding_box=bounding_box,
                lifetime=lifetime,
                speed_rate=speed_rate,
                damage_rate=damage_rate,
                penetration=penetration,
                is_explosive=is_explosive,
                explosion_radius=explosion_radius,
                explosion_image_path=explosion_image_path
            )
            
            bullet.is_active = data.get("is_active", True)
            bullet.has_collided = data.get("has_collided", False)
            
            return bullet
            
        except FileNotFoundError:
            print(f"错误: JSON文件不存在: {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"错误: JSON文件格式错误: {e}")
            return None
        except Exception as e:
            print(f"错误: 加载子弹数据时出错: {e}")
            return None
    
    def to_dict(self):
        """
        将子弹属性转换为字典（不保存到文件）
        使用示例：
            bullet = BaseBullet()
            bullet_dict = bullet.to_dict()
            print(f"子弹字典数据: {bullet_dict}")
        """
        return {
            "id": self.id,
            "shooter_id": self.shooter_id,
            "image_path": self.image_path,
            
            "position": {
                "x": self.position[0] if hasattr(self.position, '__len__') else 0.0,
                "y": self.position[1] if hasattr(self.position, '__len__') and len(self.position) > 1 else 0.0
            },
            "velocity_direction": {
                "x": self.velocity_direction[0] if hasattr(self.velocity_direction, '__len__') else 0.0,
                "y": self.velocity_direction[1] if hasattr(self.velocity_direction, '__len__') and len(self.velocity_direction) > 1 else 0.0
            },
            "size": {
                "width": self.size[0] if hasattr(self.size, '__len__') else 0.0,
                "height": self.size[1] if hasattr(self.size, '__len__') and len(self.size) > 1 else 0.0
            },
            
            "bounding_box": {
                "x": self.bounding_box.x if self.bounding_box else 0,
                "y": self.bounding_box.y if self.bounding_box else 0,
                "width": self.bounding_box.width if self.bounding_box else 0,
                "height": self.bounding_box.height if self.bounding_box else 0
            } if self.bounding_box else None,
            
            "lifetime": self.lifetime,
            "speed_rate": self.speed_rate,
            "damage_rate": self.damage_rate,
            "penetration": self.penetration,
            
            "is_explosive": self.is_explosive,
            "explosion_radius": self.explosion_radius,
            "explosion_image_path": self.explosion_image_path,
            
            "is_active": self.is_active,
            "has_collided": self.has_collided,
            
            "speed": self.speed,
            "damage": self.damage,
            "velocity": {
                "x": self.velocity[0],
                "y": self.velocity[1]
            }
        }
        
    def _update_position(self, delta_time):
        """
        根据速度向量更新位置
        """
        vx, vy = self.velocity
        x, y = self.position
        dx = vx * delta_time
        dy = vy * delta_time
        self.position = (x + dx, y + dy)
    
    def _update_bounding_box(self):
        """
        更新碰撞箱位置，等于子弹位置和大小
        """
        self.bounding_box = pygame.Rect(self.position[0], self.position[1], self.size[0], self.size[1])
        
    def update(self, delta_time):
        """
        更新子弹状态
        """
        if not self.is_active or self.has_collided:
            return False
        
        self.lifetime -= delta_time
        if self.lifetime <= 0:
            self.is_active = False
            return False

        self._update_position(delta_time)
        self._update_bounding_box()
        
        return True
    
    def set_velocity_direction(self, direction_x, direction_y):
        self.velocity_direction = (float(direction_x), float(direction_y))
        self.velocity = self.cal_velocity
        
    def set_speed_rate(self, new_speed_rate):
        self.speed_rate = float(new_speed_rate)
        self.speed = BULLET_SPEED * self.speed_rate
        self.velocity = self.cal_velocity()
        
    def move_to(self, x, y):
        self.position = (x, y)
        self._update_bounding_box()
        
    def move_by(self, dx, dy):
        x, y = self.position
        self.position = (x + dx, y + dy)
        self._update_bounding_box()
    
    def set_velocity(self, velocity_x, velocity_y):
        new_speed = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
        if new_speed > 0:
            self.velocity_direction = (velocity_x / new_speed, velocity_y / new_speed)
        else:
            self.velocity_direction = (0.0, 0.0)
        self.speed = new_speed
        self.speed_rate = self.speed / BULLET_SPEED
        self.velocity = (velocity_x, velocity_y)