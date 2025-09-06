#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器启动脚本 - Problem4-DE-Server.py
用于配置和启动80核心高性能优化

使用方法:
python scripts/server_launcher.py [options]
"""

import os
import sys
import argparse
import subprocess
import psutil
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_system_resources():
    """检查系统资源"""
    print("🔍 系统资源检查:")
    
    # CPU信息
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    print(f"  CPU核心数: {cpu_count} (物理核心: {cpu_count_physical})")
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"  内存容量: {memory_gb:.1f}GB (可用: {memory.available/(1024**3):.1f}GB)")
    
    # 磁盘空间
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    print(f"  磁盘空间: {disk_free_gb:.1f}GB 可用")
    
    # 系统建议
    print("\n💡 系统建议:")
    if cpu_count < 40:
        print(f"  ⚠️  CPU核心数({cpu_count})低于推荐值(40+)，考虑使用Problem4-DE.py")
    elif cpu_count >= 80:
        print(f"  ✅ CPU核心数充足({cpu_count})，适合80核心并行")
    else:
        print(f"  ⚡ CPU核心数({cpu_count})中等，可使用{cpu_count}核心并行")
    
    if memory_gb < 16:
        print(f"  ⚠️  内存容量({memory_gb:.1f}GB)偏低，建议16GB+")
    elif memory_gb >= 32:
        print(f"  ✅ 内存容量充足({memory_gb:.1f}GB)")
    else:
        print(f"  ⚡ 内存容量({memory_gb:.1f}GB)适中")
    
    if disk_free_gb < 5:
        print(f"  ⚠️  磁盘空间不足({disk_free_gb:.1f}GB)，检查点保存可能受限")
    
    return cpu_count, memory_gb

def setup_environment():
    """设置环境"""
    print("\n🔧 环境设置:")
    
    # 创建必要目录
    directories = ['server_checkpoints', 'logs']
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"  📁 创建目录: {dir_path}")
    
    # 检查Python依赖
    required_packages = ['numpy', 'psutil', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}: 未安装")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def launch_server_optimization(args):
    """启动服务器优化"""
    print("\n🚀 启动服务器优化...")
    
    # 构建命令
    cmd = [sys.executable, str(project_root / "Problem4-DE-Server.py")]
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 如果指定了核心数限制
    if args.max_cores:
        env['MAX_CORES'] = str(args.max_cores)
    
    # 如果指定了内存限制
    if args.max_memory:
        env['MAX_MEMORY_GB'] = str(args.max_memory)
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"工作目录: {project_root}")
    
    try:
        # 启动进程
        if args.background:
            # 后台运行
            log_file = project_root / "logs" / f"server_optimization_{int(time.time())}.log"
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, 
                    cwd=project_root,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print(f"🔄 后台运行中，PID: {process.pid}")
            print(f"📝 日志文件: {log_file}")
            print("使用 'kill -TERM {process.pid}' 优雅停止")
        else:
            # 前台运行
            subprocess.run(cmd, cwd=project_root, env=env, check=True)
            
    except KeyboardInterrupt:
        print("\n⏹️  收到中断信号，正在停止...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 优化过程出错: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="服务器版差分进化优化启动器")
    
    parser.add_argument("--max-cores", type=int, default=None,
                       help="最大使用核心数 (默认: 自动检测)")
    parser.add_argument("--max-memory", type=int, default=None,
                       help="最大使用内存(GB) (默认: 自动检测)")
    parser.add_argument("--background", "-b", action="store_true",
                       help="后台运行")
    parser.add_argument("--skip-check", action="store_true",
                       help="跳过系统检查")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🖥️  CUMCM2025 Problem 4 - 服务器启动器")
    print("="*80)
    
    # 系统检查
    if not args.skip_check:
        cpu_count, memory_gb = check_system_resources()
        
        # 根据系统资源调整参数
        if args.max_cores is None:
            args.max_cores = min(80, cpu_count)
        if args.max_memory is None:
            args.max_memory = min(32, int(memory_gb * 0.8))  # 使用80%内存
    
    # 环境设置
    if not setup_environment():
        print("\n❌ 环境设置失败，请解决依赖问题后重试")
        return 1
    
    # 最终确认
    print(f"\n📋 启动配置:")
    print(f"  最大核心数: {args.max_cores}")
    print(f"  最大内存: {args.max_memory}GB")
    print(f"  运行模式: {'后台' if args.background else '前台'}")
    
    if not args.background:
        confirm = input("\n是否开始优化？(y/n): ").lower().strip()
        if confirm != 'y':
            print("已取消启动")
            return 0
    
    # 启动优化
    success = launch_server_optimization(args)
    
    if success:
        print("\n✅ 服务器优化启动成功!")
        return 0
    else:
        print("\n❌ 服务器优化启动失败!")
        return 1

if __name__ == "__main__":
    exit(main()) 