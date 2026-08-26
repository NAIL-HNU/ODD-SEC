#!/bin/bash

# 全景相机检测系统镜像导出脚本

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

# 构建镜像
build_image() {
    print_info "构建 Docker 镜像..."
    docker build -t panorama-detection:latest .
    print_info "镜像构建完成！"
}

# 导出镜像
export_image() {
    local output_file=${1:-"panorama-detection.tar"}
    
    print_info "导出镜像到: $output_file"
    docker save panorama-detection:latest > "$output_file"
    print_info "镜像导出完成！文件大小: $(du -h "$output_file" | cut -f1)"
}

# 显示帮助
show_help() {
    echo "全景相机检测系统镜像导出脚本"
    echo ""
    echo "用法: $0 [选项] [输出文件名]"
    echo ""
    echo "选项:"
    echo "  build     构建镜像"
    echo "  export    导出镜像"
    echo "  all       构建并导出 (推荐)"
    echo "  help      显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 all                           # 构建并导出到 panorama-detection.tar"
    echo "  $0 all my-custom-name.tar       # 构建并导出到 my-custom-name.tar"
    echo "  $0 build                         # 仅构建镜像"
    echo "  $0 export                        # 仅导出镜像"
}

# 主函数
main() {
    local command=$1
    local output_file=$2
    
    case $command in
        "build")
            check_docker
            build_image
            ;;
        "export")
            check_docker
            export_image "$output_file"
            ;;
        "all")
            check_docker
            build_image
            export_image "$output_file"
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