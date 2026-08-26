#!/bin/bash

# 全景相机检测系统 Docker 运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker 服务未运行，请启动 Docker 服务"
        exit 1
    fi
}

# 检查Docker Compose是否安装
check_docker_compose() {
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
}

# 检查NVIDIA Docker (可选)
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null; then
        print_info "检测到 NVIDIA Docker，将使用GPU支持"
        USE_GPU=true
    else
        print_warning "未检测到 NVIDIA Docker，将使用CPU模式"
        USE_GPU=false
    fi
}

# 构建镜像
build_image() {
    print_info "构建 Docker 镜像..."
    docker compose build
}

# 运行容器
run_container() {
    local service_name=$1
    
    print_info "启动容器: $service_name"
    
    if [ "$USE_GPU" = true ]; then
        docker compose up -d panorama-detection-gpu
        docker compose exec panorama-detection-gpu bash
    else
        docker compose up -d panorama-detection
        docker compose exec panorama-detection bash
    fi
}

# 停止容器
stop_container() {
    print_info "停止容器..."
    docker compose down
}

# 清理容器和镜像
cleanup() {
    print_info "清理容器和镜像..."
    docker compose down --rmi all --volumes --remove-orphans
}

# 显示帮助信息
show_help() {
    echo "全景相机检测系统 Docker 管理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build     构建 Docker 镜像"
    echo "  run       运行容器 (交互模式)"
    echo "  start     启动容器 (后台模式)"
    echo "  stop      停止容器"
    echo "  restart   重启容器"
    echo "  clean     清理容器和镜像"
    echo "  help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build    # 构建镜像"
    echo "  $0 run      # 运行容器并进入bash"
    echo "  $0 stop     # 停止容器"
}

# 主函数
main() {
    local command=$1
    
    case $command in
        "build")
            check_docker
            check_docker_compose
            build_image
            ;;
        "run")
            check_docker
            check_docker_compose
            check_nvidia_docker
            run_container
            ;;
        "start")
            check_docker
            check_docker_compose
            check_nvidia_docker
            if [ "$USE_GPU" = true ]; then
                docker compose up -d panorama-detection-gpu
            else
                docker compose up -d panorama-detection
            fi
            print_info "容器已在后台启动"
            ;;
        "stop")
            stop_container
            ;;
        "restart")
            stop_container
            sleep 2
            if [ "$USE_GPU" = true ]; then
                docker compose up -d panorama-detection-gpu
            else
                docker compose up -d panorama-detection
            fi
            print_info "容器已重启"
            ;;
        "clean")
            cleanup
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

# 运行主函数
main "$@" 