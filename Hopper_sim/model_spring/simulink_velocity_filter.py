


import numpy as np

class SimulinkVelocityFilter:
    
    def __init__(self, dt=0.0001, forgetting_factor=0.4):
        self.dt = dt
        self.forgetting_factor = forgetting_factor
        

        self.pos_prev = None
        

        self.ema_state = 0.0
        
    def update(self, motor_pos):

        if self.pos_prev is None:
            self.pos_prev = motor_pos
            return 0.0
        
        raw_velocity = (motor_pos - self.pos_prev) / self.dt
        self.pos_prev = motor_pos
        


        self.ema_state = (self.forgetting_factor * self.ema_state + 
                         (1 - self.forgetting_factor) * raw_velocity)
        
        return self.ema_state
    
    def reset(self):
        self.pos_prev = None
        self.ema_state = 0.0


class SimulinkVelocityFilterVector:
    
    def __init__(self, dt=0.001, forgetting_factor=0.98):

        self.filters = [
            SimulinkVelocityFilter(dt, forgetting_factor),
            SimulinkVelocityFilter(dt, forgetting_factor),
            SimulinkVelocityFilter(dt, forgetting_factor)
        ]
    
    def update(self, q_curr):
        qd_filtered = np.zeros(3)
        for i in range(3):
            qd_filtered[i] = self.filters[i].update(q_curr[i])
        return qd_filtered
    
    def reset(self):
        for f in self.filters:
            f.reset()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Simulink Joint Position Filter 测试")
    print("基于指数加权移动平均 (Exponential Weighting)")
    print("="*60)
    

    dt = 0.001
    t = np.arange(0, 5, dt)
    

    velocity_true = 2 * np.sin(t * 2 * np.pi * 0.5)
    velocity_true[t > 2.5] += 1.5
    

    position_true = np.cumsum(velocity_true) * dt
    

    noise_level = 0.05
    position_measured = position_true + np.random.randn(len(t)) * noise_level
    

    simulink_filter = SimulinkVelocityFilter(dt=dt, forgetting_factor=0.95)
    
    velocity_simulink = []
    for p in position_measured:
        v_est = simulink_filter.update(p)
        velocity_simulink.append(v_est)
    

    velocity_diff = np.diff(position_measured, prepend=position_measured[0]) / dt
    

    plt.figure(figsize=(14, 8))
    plt.plot(t, velocity_true, 'g-', linewidth=3, label='True Velocity (真实速度)')
    plt.plot(t, velocity_diff, 'r-', alpha=0.3, label='Simple Derivative (简单差分)')
    plt.plot(t, velocity_simulink, 'b-', linewidth=2, 
             label=f'Simulink Filter (λ={simulink_filter.forgetting_factor}, 3路求和)')
    
    plt.title('Simulink Joint Position Filter - Exponential Weighting (1000Hz)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity (rad/s)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/abc/Hopper/simulink_filter_test.png', dpi=150)
    print("\n✅ 图像已保存到: /home/abc/Hopper/simulink_filter_test.png")
    plt.show()
    

    start_idx = 100
    error_simulink = np.abs(np.array(velocity_simulink[start_idx:]) - velocity_true[start_idx:])
    error_diff = np.abs(velocity_diff[start_idx:] - velocity_true[start_idx:])
    
    print(f"\n性能对比 (去掉前{start_idx}个初始化点):")
    print(f"Simulink滤波 - 平均误差: {np.mean(error_simulink):.4f}, 最大误差: {np.max(error_simulink):.4f}")
    print(f"简单差分    - 平均误差: {np.mean(error_diff):.4f}, 最大误差: {np.max(error_diff):.4f}")
    
    if np.mean(error_diff) > 0:
        improvement = (1 - np.mean(error_simulink)/np.mean(error_diff))*100
        print(f"Simulink改善: {improvement:.1f}%")
    
    print("\n" + "="*60)
    print("模型说明:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("流程: motor_pos → 离散微分 → 分三路 → EMA × λ → Mux求和")
    print(f"      最终输出 = 3 × λ × EMA")
    print(f"      λ={simulink_filter.lambda_factor}, α={simulink_filter.alpha}")
    print("="*60)
