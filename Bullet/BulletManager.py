"""子弹管理器，负责管理所有子弹的更新和绘制"""
from Unit.Tank.Tank import *
from Map.Map import *
from GameMode import *
from Bullet.NormalShell.NormalShell import *
from Unit.EnemyAI import *

class BulletManager:

    def __init__(self):
        self.bullets = []  # 存储所有活跃的子弹
        self.to_remove = []  # 存储待移除的子弹
        
    def add_bullet(self, bullet):
        if bullet:
            self.bullets.append(bullet)
            if BULLET_INFO_TEXT or DEBUG_MODE:
                print(f"子弹发射: ID={bullet.id}, 位置={bullet.position}")
    
    def update(self, delta_time, units, obstacles):
        self.to_remove.clear()
        
        for bullet in self.bullets:
            # 更新子弹状态
            if not bullet.update(delta_time, units, obstacles):
                # 子弹不再活跃，标记为待移除
                if bullet in self.bullets and bullet not in self.to_remove:
                    self.to_remove.append(bullet)
            
            # 如果子弹爆炸了，应用爆炸伤害
            if bullet.has_exploded and bullet.is_explosive:
                bullet.apply_explosion_damage(units)
        
        # 移除不再活跃的子弹
        for bullet in self.to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
                if BULLET_INFO_TEXT or DEBUG_MODE:
                    print(f"子弹移除: ID={bullet.id}")
    
    def draw(self, surface, camera_offset):
        """绘制所有子弹"""
        for bullet in self.bullets:
            bullet.draw(surface, camera_offset)
    
    def get_active_count(self):
        """获取活跃子弹数量"""
        return len([b for b in self.bullets if b.is_active])
    
    def clear(self):
        """清空所有子弹"""
        self.bullets.clear()

