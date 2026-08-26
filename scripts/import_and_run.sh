#!/bin/bash

# 全景相机检测系统镜像导入和运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker 服务未运行"
        exit 1
    fi
}

# 导入镜像
import_image() {
    local tar_file=${1:-"panorama-detection.tar"}
    
    if [ ! -f "$tar_file" ]; then
        print_error "镜像文件不存在: $tar_file"
        exit 1
    fi
    
    print_info "导入镜像: $tar_file"
    docker load < "$tar_file"
    print_info "镜像导入完成！"
}

# 运行系统
run_system() {
    print_info "启动全景相机检测系统..."
    
    # 检查模型文件
    if [ ! -d "model" ]; then
        print_warning "未找到 model 目录，请确保模型文件已准备"
    fi
    
    # 运行容器
    docker run -it --rm \
        -v $(pwd)/src:/catkin_ws/src \
        -v $(pwd)/launch:/catkin_ws/launch \
        -v $(pwd)/config:/catkin_ws/config \
        -v $(pwd)/model:/catkin_ws/model \
        -v $(pwd)/data:/catkin_ws/data \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --network host \
        --privileged \
        panorama-detection:latest
}

# 显示帮助
show_help() {
    echo "全景相机检测系统镜像导入和运行脚本"
    echo ""
    echo "用法: $0 [选项] [tar文件名]"
    echo ""
    echo "选项:"
    echo "  import    导入镜像"
    echo "  run       运行系统"
    echo "  deploy    导入并运行 (推荐)"
    echo "  help      显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 deploy                    # 导入 panorama-detection.tar 并运行"
    echo "  $0 deploy my-image.tar      # 导入 my-image.tar 并运行"
    echo "  $0 import                    # 仅导入镜像"
    echo "  $0 run                       # 仅运行系统"
}

# 主函数
main() {
    local command=$1
    local tar_file=$2
    
    case $command in
        "import")
            check_docker
            import_image "$tar_file"
            ;;
        "run")
            check_docker
            run_system
            ;;
        "deploy")
            check_docker
            import_image "$tar_file"
            run_system
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@" 