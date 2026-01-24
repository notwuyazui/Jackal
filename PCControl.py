import pygame

def PC():
        # 相机偏移
    camera_offset = [0, 0]
    camera_speed = 5
    
    # 控制状态
    moving_forward = False
    moving_backward = False
    turning_left = False
    turning_right = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_w:
                moving_forward = True
                print("W键按下: 前进")
            elif event.key == pygame.K_s:
                moving_backward = True
                print("S键按下: 后退")
            elif event.key == pygame.K_a:
                turning_left = True
                print("A键按下: 左转")
            elif event.key == pygame.K_d:
                turning_right = True
                print("D键按下: 右转")
            elif event.key == pygame.K_F1:
                debug_mode = not debug_mode
                print(f"调试模式: {'开启' if debug_mode else '关闭'}")
        
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                moving_forward = False
                print("W键释放")
            elif event.key == pygame.K_s:
                moving_backward = False
                print("S键释放")
            elif event.key == pygame.K_a:
                turning_left = False
                print("A键释放")
            elif event.key == pygame.K_d:
                turning_right = False
                print("D键释放")

    # 处理相机移动
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_offset[0] -= camera_speed
    if keys[pygame.K_RIGHT]:
        camera_offset[0] += camera_speed
    if keys[pygame.K_UP]:
        camera_offset[1] -= camera_speed
    if keys[pygame.K_DOWN]:
        camera_offset[1] += camera_speed