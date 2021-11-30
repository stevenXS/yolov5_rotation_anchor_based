# -*- coding:utf-8 -*-
# @Time: 2021/11/2517:24
# @Author: StevenX
# @Description:

# !/usr/bin/env python3

# -*- coding: utf-8 -*-


import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


class SendResultByEmail(object):
    def __init__(self, receiver, title, content):
        self.sender = receiver  # 发送地址
        self.title = title  # 标题
        self.content = content  # 发送内容
        self.sys_sender = 'x1165160978@163.com'  # 系统账户
        self.sys_pwd = 'WACXRPRKPQCJSXTF'  # 系统账户密码

    def send(self):
        try:
            """
            构造一个邮件对象,
            第一个参数就是邮件正文，
            第二个参数是MIME的subtype，传入'html'，最终的MIME就是'text/html'。
            最后一定要用utf-8编码保证多语言兼容性。
            """
            msg = MIMEText(self.content, 'html', 'utf-8')
            # 发件人格式
            msg['From'] = formataddr(["", self.sys_sender])
            # 收件人格式
            msg['To'] = formataddr(["", self.sender])
            # 邮件主题
            msg['Subject'] = self.title
            # SMTP服务器
            client = smtplib.SMTP()
            client.connect("smtp.163.com", 25)
            # 登录账户
            client.login(self.sys_sender, self.sys_pwd)
            # 发送邮件
            client.sendmail(self.sys_sender, [self.sender, ], msg.as_string())
            # 退出账户
            client.quit()
            return True
        except Exception as e:
            print(e)
            return False


if __name__ == '__main__':


    # receiver = 'steven@stu.scu.edu.cn'  # 发送地址
    receiver = 'x1165160978@163.com'  # 发送地址
    title = 'DOTA数据集测试结果'  # 标题
    # content = 'mAP: 50%'  # 发送内容
    content = '测试' # 发送内容

    # 调用send方法，发送邮件
    ret = SendResultByEmail(receiver, title, content).send()
    if ret:
        print('发送成功!')
    else:
        print('发送失败!')

