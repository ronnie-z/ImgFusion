import logging

#编程的方式记录日志

def get_format():
    format_str = '[%(asctime)s- %(levelname)-9s %(filename)-8s # %(lineno)-3d line] - %(message)s'
    formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %H:%M:%S")
    return formatter

def logger_init():
    #记录器
    logger1 = logging.getLogger("global")
    logger1.setLevel(logging.INFO)



    #处理器
    #1.标准输出  负责控制台输出
    sh1 = logging.StreamHandler()
    sh1.setLevel(logging.INFO)
    sh1.setFormatter(get_format())


    # 2.文件输出  负责文件输出
    # 没有设置输出级别，将用logger1的输出级别(并且输出级别在设置的时候级别不能比Logger的低!!!)
    fh1 = logging.FileHandler(filename="log_second_train.txt")
    fh1.setFormatter(get_format())

    # 格式器
    fmt1 = logging.Formatter(fmt="%(asctime)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s")

    fmt2 = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                            ,datefmt="%Y/%m/%d %H:%M:%S")

    #给处理器设置格式
    # sh1.setFormatter(fmt1)
    # fh1.setFormatter(fmt2)

    #记录器设置处理器
    logger1.addHandler(sh1)
    logger1.addHandler(fh1)

#打印日志代码
# logger1.debug("This is  DEBUG of logger1 !!")
# logger1.info("This is  INFO of logger1 !!")
# logger1.warning("This is  WARNING of logger1 !!")
# logger1.error("This is  ERROR of logger1 !!")
# logger1.critical("This is  CRITICAL of logger1 !!")
