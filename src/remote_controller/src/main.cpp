#include <iostream>
#include <communication/msg/motion_commands.hpp>
#include <linux/joystick.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using namespace std;

#if 1  //PS4 JS
#define JS_VELX_AXIS 4
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1

#define JS_STOP_BT 10
#define JS_START_BT 9

#define JS_SHOULDER_LEFT_BT  6
#define JS_SHOULDER_RIGHT_BT  7
#define JS_A_BT 0
#define JS_B_BT 1
#define JS_X_BT 2
#define JS_Y_BT 3

// #define JS_GAIT_STAND_BT 0
// #define JS_GAIT_WALK_BT 2
// #define JS_HEIGHT_UPPER_BT  1
// #define JS_HEIGHT_LOWER_BT  3
// #define JS_MODE_BT          5
#else //XBOX JS
#define JS_VELX_AXIS 3
#define JS_VELX_AXIS_DIR -1
#define JS_VELY_AXIS 0
#define JS_VELY_AXIS_DIR -1
#define JS_VELR_AXIS 6
#define JS_VELR_AXIS_DIR -1

#define JS_STOP_BT 11
#define JS_START_BT 13

#define JS_SHOULDER_LEFT_BT  6
#define JS_SHOULDER_RIGHT_BT  7
#define JS_Y_BT 4
#define JS_X_BT 3
#define JS_A_BT 0
#define JS_B_BT 1
#endif

#define AXIS_DEAD_ZONE  1000

// #define MIN_SPEED_X -1.0
#define MIN_SPEED_X -2.0
// #define MAX_SPEED_X 5.0
#define MAX_SPEED_X 8.0
#define MIN_SPEED_Y -0.6
#define MAX_SPEED_Y 0.6
// #define MIN_SPEED_Y -1.0
// #define MAX_SPEED_Y 1.0
#define MIN_SPEED_R -1.5
#define MAX_SPEED_R 1.5

#define AXIS_VALUE_MAX 32767

#define STAND_HEIGHT 1.0
#define STAND_HEIGHT_MIN    1.0
#define STAND_HEIGHT_MAX    3.0

class COMPublisher : public rclcpp::Node{
public:
    COMPublisher(const char *_js_dev) : Node("COM_publisher"){
        if (strlen(_js_dev) >= 128){
            printf("dev:%s error\n", _js_dev);
            exit(-1);
        }

        strcpy(_js_dev_name, _js_dev);
        
        while (1){
            js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
            if (js_fd < 0){
                printf("open:%s failed\n", _js_dev_name);
                sleep(1);      
            }
            else{
                printf("open js dev: %s\n", _js_dev_name);
                break;
            }
        }
        
        com_pub = this->create_publisher<communication::msg::MotionCommands>("motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));
        js_loop_thread_ = std::thread(&COMPublisher::js_loop, this);
    }

    ~COMPublisher(){
        if (js_fd > 0){
            close(js_fd);
        }
    }

private:
    mutable std::mutex lock_;

    char _js_dev_name[128] = {0};
    int js_fd;
    double js_axis[20] = {0};   //原始数据
    double js_bt[20] = {0};
    std::thread js_loop_thread_;

    double velxy[2] = {0};      //x y速度
    double velxy_filt[2] = {0}; //x y速度滤波值
    double stand_height = STAND_HEIGHT;
    double height_filt = STAND_HEIGHT;
    double velr = 0;    //旋转速度
    double velr_filt = 0;
    double vel_offset = 0.0;

    bool model_y = false;  // (Y)
    bool model_x = false;  // (X)
    bool model_b = false;  // (B)
    bool model_a = false;  // (A)

    void timer_callback(){
        auto message = communication::msg::MotionCommands();{
            const std::lock_guard<std::mutex> guard(lock_);

            velxy[0] = (js_axis[JS_VELX_AXIS] * JS_VELX_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velxy[1] = (js_axis[JS_VELY_AXIS] * JS_VELY_AXIS_DIR) / (double)AXIS_VALUE_MAX;
            velr = (js_axis[JS_VELR_AXIS] * JS_VELR_AXIS_DIR) / (double)AXIS_VALUE_MAX;

            velxy[0] = fabs(velxy[0]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[0] : 0;
            velxy[1] = fabs(velxy[1]) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velxy[1] : 0;
            velr = fabs(velr) > AXIS_DEAD_ZONE / (double)AXIS_VALUE_MAX ? velr : 0;
            
            //按定义最大速度缩放
            if (velxy[0] > 0){
                velxy[0] *= MAX_SPEED_X;
            }
            else if (velxy[0] < 0){
                velxy[0] *= -MIN_SPEED_X;
            }

            if (velxy[1] > 0){
                velxy[1] *= MAX_SPEED_Y;
            }
            else if (velxy[1] < 0){
                velxy[1] *= -MIN_SPEED_Y;
            }

            if (velr > 0){
                velr *= MAX_SPEED_R;
            }
            else if (velr < 0){
                velr *= -MIN_SPEED_R;
            }

            velxy_filt[0] = velxy[0] * 0.03 + velxy_filt[0] * 0.97;
            velxy_filt[1] = velxy[1] * 0.03 + velxy_filt[1] * 0.97;

            velr_filt = velr * 0.05 + velr_filt *  0.95;

            message.vel_des.x = velxy_filt[0] + vel_offset;
            message.vel_des.y = velxy_filt[1];
            message.yawdot_des = velr_filt;

            // 设置手臂控制标志
            message.btn_5 = model_a ? 1 : 0; // A
            message.btn_6 = model_x ? 1 : 0; // X
            message.btn_7 = model_y ? 1 : 0; // Y
            message.btn_10 = model_b ? 1 : 0; // B

            height_filt = height_filt * 0.9 + stand_height * 0.1;
            message.height_des = height_filt;
        }

        com_pub->publish(message);
    }

    void reset_value()
    {
        const std::lock_guard<std::mutex> guard(lock_);
        memset(js_axis, 0, sizeof(js_axis));
        memset(velxy, 0, sizeof(velxy));
        memset(velxy_filt, 0, sizeof(velxy_filt));
        velr_filt = 0;
        height_filt = STAND_HEIGHT;
    }

    void js_loop(){
        while (1){
            ssize_t len;
            struct js_event event;

            len = read(js_fd, &event, sizeof(event));

            if (len == sizeof(event)){
                if (event.type & JS_EVENT_AXIS){
                    //printf("Axis: %d -> %d\n", (int)event.number, (int)event.value);
                    js_axis[event.number] = event.value;
                }
                else if (event.type & JS_EVENT_BUTTON){
                    //printf("Button: %d -> %d\n", (int)event.number, (int)event.value);
                    if (event.value){
                        switch (event.number){
                        case JS_STOP_BT:{
                            // 杀掉关节可视化进程
                            system("ps aux | grep 'python3 src/bxi_example_py_elf3/bxi_example_py_elf3/joint_tn_visualizer.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9");

                            system("killall -9 simulation");// mujoco
                            system("killall -9 bxi_example_py_elf3_dance");

                            system("killall -SIGINT hardware_elf3");
                            system("killall -SIGINT bxi_example_py_elf3");
                            system("killall -SIGINT bxi_example_py_elf3_dance_hw");

                            reset_value();
                        }
                        break;
                        case JS_START_BT:{
                            // 确保日志目录存在（使用 -p 避免目录已存在时报错）
                            // 根据权限选择 launch 文件
                            std::string launch_cmd;
                            if (geteuid() == 0) {  // root 权限（例如 sudo su 下）
                                system("mkdir -p ./log/bxi_real_log");
                                launch_cmd = "ros2 launch bxi_example_py_elf3 example_dance_hw.launch.py > ./log/bxi_real_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log 2>&1 &";
                                printf("Running hardware launch (root)\n");
                                system(launch_cmd.c_str());
                            } else {
                                system("mkdir -p ./log/bxi_sim_log");
                                // 启动关节可视化（先启动）
                                system("nohup python3 src/bxi_example_py_elf3/bxi_example_py_elf3/joint_tn_visualizer.py > /dev/null 2>&1 &");
                                // 等待1秒，确保可视化先启动
                                sleep(1);
                                // 再启动仿真
                                launch_cmd = "ros2 launch bxi_example_py_elf3 example_dance.launch.py > ./log/bxi_sim_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log 2>&1 &";
                                printf("Running simulation launch (non-root)\n");
                                system(launch_cmd.c_str());
                            }
                            reset_value();
                        }
                        break;
                        case JS_SHOULDER_LEFT_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            stand_height -= 0.2;
                            if (stand_height > STAND_HEIGHT_MAX)
                            {
                                stand_height = STAND_HEIGHT_MAX;
                            }
                            printf("stand_height: %f\n", stand_height);
                        }
                        break;
                        case JS_SHOULDER_RIGHT_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            stand_height += 0.2;
                            if (stand_height < STAND_HEIGHT_MIN)
                            {
                                stand_height = STAND_HEIGHT_MIN;
                            }
                            printf("stand_height: %f\n", stand_height);
                        }
                        break;
                        case JS_A_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            model_a = !model_a;
                            printf("bt_a\n");
                        }
                        break;
                        case JS_X_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            model_x = !model_x;
                            printf("bt_x\n");
                        }
                        break;
                        case JS_Y_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            model_y = !model_y;
                            printf("bt_y\n");
                        }
                        break;
                        case JS_B_BT:{
                            const std::lock_guard<std::mutex> guard(lock_);
                            model_b = !model_b;
                            printf("bt_b\n");
                        }
                        break;
                        default:
                            break;
                        }
                    }
                }
                else{
                    printf("unknown event:%u\n", event.type);
                }
            }
            if (len <= 0){
                printf("js dev lost, retry\n");
                close(js_fd);
                while (1){
                    js_fd = open(_js_dev_name, O_RDONLY); // O_NONBLOCK
                    if (js_fd < 0){
                        printf("open:%s failed\n", _js_dev_name);
                        sleep(1);
                    }
                    else{
                        printf("open js dev: %s\n", _js_dev_name);
                        break;
                    }
                }
            }
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub;
};

int main(int argc, const char *argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>("/dev/input/js0"));
    rclcpp::shutdown();

    return 0;
}
