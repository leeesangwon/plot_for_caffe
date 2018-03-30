from __future__ import print_function 

import sys, os
import re, argparse, subprocess
from datetime import datetime, timedelta
import logging

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import style

# for slackbot
CHANNEL = 'change to your channel'
BOT_TOKEN = "change to your bot token"

LOSS_MIN = 0.1
LOSS_MAX = 0.3
STEP_MAX = 90000
LR_MULT = 10000 # learning rate multiplyer

logger = logging.getLogger('')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)


class Subplot(object):
    def __init__(self, ax=None, fig=None, 
                 x_mult=1, y_mult=1, x_regex=None, y_regex=None, 
                 xlabel=None, ylabel=None, 
                 xmin=0, xmax=None, ymin=0, ymax=None):
        if ax is not None:
            if fig is not None:
                raise ValueError('ax and fig cannot be used together')
            self.ax = ax.twinx()
        else:
            if fig is None:
                raise ValueError('ax or fig is needed')
            self.ax = fig.add_subplot(1, 1, 1)
        if x_regex is None:
            raise ValueError('x_regex is needed')
        if y_regex is None:
            raise ValueError('y_regex is needed')
        self.x_mult = x_mult
        self.y_mult = y_mult
        self.ylabel = ylabel
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            if y_mult != 1:
                ylabel = ylabel + " (x%s)" % y_mult
            self.ax.set_ylabel(ylabel)
        if xmax is not None:
            self.ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            self.ax.set_ylim(ymin, ymax)

        self.x_regex = x_regex
        self.y_regex = y_regex

        self.auxiliary_line_dict = dict()
    
    def init_target_line(self, log_file_path, line_type=None, label_header=''):
        if type(label_header) is not str:
            raise ValueError('label_header should be str type') 
        label = label_header + self.ylabel
        plot_, = self.ax.plot([], [], line_type, label=label)
        self.target_line = Line(plot_, log_file_path, self.x_regex, self.y_regex, self.x_mult, self.y_mult, line_type, label)
        self.target_label = label

    def add_auxiliary_line(self, log_file_path, line_type='--', label_header=''):
        if type(label_header) is not str:
            raise ValueError('label_header should be str type') 
        label = label_header + self.ylabel
        plot_, = self.ax.plot([], [], line_type, label=label)
        self.auxiliary_line_dict[label] = Line(plot_, log_file_path, self.x_regex, self.y_regex, self.x_mult, self.y_mult, line_type, label)
    
    def get_all_lines(self):
        lines = [self.target_line] + self.auxiliary_line_dict.values()
        return [x.plot for x in lines]
    
    def get_all_labels(self):
        return [self.target_label] + self.auxiliary_line_dict.keys()


class Line(object):
    def __init__(self, plot, log_file_path, x_regex, y_regex, x_mult=1, y_mult=1, line_type=None, label=None):
        with open(log_file_path, 'r') as log_file:
            logs = log_file.read()
        self.logs = logs
        self.x_regex = re.compile(x_regex)
        self.y_regex = re.compile(y_regex)
        self.x_mult = x_mult
        self.y_mult = y_mult
        self.line_type = line_type
        self.label = label
        self.plot = plot

    def draw(self):
        x_list = self.__parse_to_list(self.logs, self.x_regex)
        y_list = self.__parse_to_list(self.logs, self.y_regex)
        x_list = [x*self.x_mult for x in x_list]
        y_list = [y*self.y_mult for y in y_list]
        assert len(x_list) == len(y_list), "length of x(%s) != length of y(%s)" % (len(x_list), len(y_list))
        self.plot.set_data(x_list, y_list)
    
    def __parse_to_list(self, logs, regex):
        regex_iter = regex.finditer(logs)
        return [self.__line_to_float(x.group()) for x in regex_iter]

    def __line_to_float(self, text):
        re_float_number = re.compile(r"[\d.]+(e[-+][\d]{2}){0,1}")
        return float(re_float_number.search(text).group())


class SlackHandler(object):
    def __init__(self, slack_alert, bot_token, channel):
        self.is_active = bool(slack_alert)
        self.alert_interval = slack_alert
        self.bot_token = bot_token
        self.channel = channel
        self.last_fid = None

    def deactivate(self):
        self.is_active = False

    def send_message(self, title, body):
        def _send_message(title, body):
            try:
                result = subprocess.check_output("curl -F token=%s -F channel=#%s -F text=\"*%s* %s\" https://slack.com/api/chat.postMessage"
                                            % (self.bot_token, self.channel, title, body), stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot send message to slack")
            if not re.search(r'"ok":true', result):
                logger.warning("cannot send message to slack")
            return result
        result = _send_message(title, body)

    def send_figure(self, fig, graph_title):
        temp_figure_image_name = "tmp_fig.png"
        fig.savefig(temp_figure_image_name)
        try:
            self.send_image(graph_title, temp_figure_image_name)
        except AssertionError as e:
            print(e)
        os.remove(temp_figure_image_name)

    def send_image(self, title, image_path):
        def _send_image(title, image_path):
            try:
                result = subprocess.check_output("curl -F token=%s -F channels=#%s -F title=%s -F file=@%s https://slack.com/api/files.upload"
                                            % (self.bot_token, self.channel, title, image_path), stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot send image to slack")
            if not re.search(r'"ok":true', result):
                logger.warning("cannot send image to slack")
            return result
        
        def _delete_last_image(fid):
            try:
                result = subprocess.check_output("curl -F token=%s -F file=%s https://slack.com/api/files.delete" % (self.bot_token, fid), shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot delete last image")
            if not re.search(r'"ok":true', result):
                logger.warning("cannot delete last image: '%s'" % result)

        def _get_file_id(result):
            targetline = re.search(r'"id":"\w+"', result).group()
            return re.findall(r'\w+', targetline)[-1]
        
        result = _send_image(title, image_path)
        if self.last_fid is not None:
            _delete_last_image(self.last_fid)
        self.last_fid = _get_file_id(result)


class LineColorCycler(object):
    def __init__(self):
        self.__num_of_colors = 10
        self.__color_list = ['C%s' % i for i in range(self.__num_of_colors)]
        self.__current_index = 1

    def __call__(self):
        return self.__color_list[self.__get_current_index()]

    def __get_current_index(self):
        current_index = self.__current_index
        self.__current_index = (self.__current_index + 1) % self.__num_of_colors
        return current_index


line_color_cycler = LineColorCycler()


def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("log_file_path", type=str, help="path to the log file to plot")
    PARSER.add_argument("-c", "--comparision", type=str, default=None, help="path to the comparision log file to plot, default: not plot")
    PARSER.add_argument("-g", "--graph_title", type=str, default=None, help="title of the graph, default: the log file's name")
    PARSER.add_argument("-i", "--interval", type=int, default=10000, help="plotting interval(ms), default: 10000")
    PARSER.add_argument("-d", "--dark_mode", action="store_true", help="turns on the dark mode")
    PARSER.add_argument("-q", "--auto_quit", action="store_true", help="quit automatically after optimization done")
    PARSER.add_argument("-s", "--slack_alert", type=int, default=0, help="send message after optimization done or figure every SLACK_ALERT minutes to slack, default is 0(not send)")
    ARGS = PARSER.parse_args()
    
    if not os.path.isfile(ARGS.log_file_path):
        logger.error("there is no log file at '%s'" % ARGS.log_file_path)
        raise TypeError
    graph_title = ARGS.graph_title
    if graph_title is None:
        graph_title = os.path.basename(ARGS.log_file_path).replace(".log", "")
    if ARGS.dark_mode:
        style.use('dark_background')
    
    plot(ARGS.log_file_path, ARGS.comparision, graph_title, ARGS.slack_alert, ARGS.auto_quit, ARGS.interval)


def plot(log_file_path, comparision_log_file_path, graph_title, slack_alert, auto_quit, refresh_interval, lr_mult=LR_MULT):
    fig = plt.figure()
    subplot_dict = dict()
    subplot_dict['loss'] = Subplot(fig=fig, xlabel="iteration", xmax=STEP_MAX, ylabel="loss", ymin=LOSS_MIN, ymax=LOSS_MAX, 
                            x_regex=r"Iteration [\d]+, loss", y_regex=r", loss = [\d.]+(e[-+][\d]{2}){0,1}")
    subplot_dict['lr'] = Subplot(ax=subplot_dict['loss'].ax, y_mult=lr_mult, ylabel='lr', ymin=0, ymax=1, 
                            x_regex=r"Iteration [\d]+, lr", y_regex=r"lr = [\d.]+(e[-+][\d]{2}){0,1}")
    
    for _subplot in subplot_dict.values():
        _subplot.init_target_line(log_file_path=log_file_path, line_type=line_color_cycler())

    plot_comparision(subplot_dict, comparision_log_file_path, label_header="comp_")
    setup_legend(subplot_dict)
    slack_handler = SlackHandler(slack_alert, BOT_TOKEN, CHANNEL)

    def plot_iteratively():
        def animate(frame):
            for _subplot in subplot_dict.values():
                _subplot.target_line.draw()
            if slack_handler.is_active and n_minutes_timer(slack_handler.alert_interval):
                fig = plt.gcf()
                slack_handler.send_figure(fig, graph_title)
            if is_optimization_done(log_file_path):
                if slack_handler.is_active:
                    fig = plt.gcf()
                    slack_handler.send_figure(fig, graph_title)
                    slack_handler.send_message(graph_title, 'optimization done')
                    slack_handler.deactivate()
                if auto_quit:
                    sys.exit()

        return animation.FuncAnimation(fig, animate, interval=refresh_interval)
    
    ani = plot_iteratively()
    
    plt.title(graph_title, fontsize=15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.85)
    plt.show()


def plot_comparision(subplot_dict, comparision_log_file_path, label_header):
    try:
        assert os.path.isfile(comparision_log_file_path), "no comparision file at '%s'" % comparision_log_file_path        
    except AssertionError as e:
        logger.warning(e)
    except TypeError: # no comparision option
        pass
    else:
        for _subplot in subplot_dict.values():
            _subplot.add_auxiliary_line(log_file_path=comparision_log_file_path, line_type=line_color_cycler() + '--', label_header=label_header)
            for line in _subplot.auxiliary_line_dict.values():
                line.draw()


def setup_legend(subplot_dict):
    lines, labels = [], []
    for _subplot in subplot_dict.values():
        lines += _subplot.get_all_lines()
        labels += _subplot.get_all_labels() 
    if len(lines) != len(labels):
        logger.error("the number of lines to plot and labels are different, # of lines: '%s', # of labels: '%s'" % (len(lines), len(labels)))
        raise AssertionError
    plt.legend(lines, labels)


def is_optimization_done(log_file_path):
    with open(log_file_path, 'r') as log_file:
        logs = log_file.read()
    re_optimization_done = re.compile(r"Optimization Done.")
    if re_optimization_done.search(logs):
        return True
    return False


def n_minutes_timer(n):
    try:
        is_time_to_go = n_minutes_timer.time_to_go < datetime.now()
    except AttributeError:
        n_minutes_timer.time_to_go = datetime.now()
        is_time_to_go = True
    if is_time_to_go:
        n_minutes_timer.time_to_go += timedelta(minutes=n)
    return is_time_to_go


if __name__=="__main__":
    main()
