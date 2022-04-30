# coding: utf-8

import sys
import random

class PreProcess:
    def __init__(self, conf_file='', is_test=False):
        self.conf = {}
        self.is_test = is_test
        self.r = 0.0
        self.need_normalize = False
        self.need_filt_slot = False
        self.filt_slot_set = set()
        self.zs_dic = {}

        if conf_file != '':
            self.load_conf(conf_file)
            self.init_conf()

    def g(self, ss):
        if ss in self.conf:
            return self.conf[ss]
        return ''

    def neg_sample(self, rate):
        r = random.random()
        if r <= rate:
            return True
        return False

    def load_filtslot(self, fpath):
        slot_dic = set()
        f = open(path)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            lis = line.split(':')
            slot = int(lis[0])
            if len(lis) > 1:
                span = int(lis[1])
                for i in xrange(span):
                    si = slot + i
                    slot_dic.add(si)
            else:
                slot_dic.add(slot)
        f.close()
        return slot_dic

    def load_zscore(self, fpath):
        zs_dic = {}
        f = open(fpath)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            lis = line.split(':')
            if len(lis) != 3:
                continue
            slot = int(lis[0])
            zs_dic[slot] = (float(lis[1]), float(lis[2]))
        f.close()
        return zs_dic

    def normal(self, slot, val, zs_dic):
        res = val
        if slot in zs_dic:
            zs = zs_dic[slot]
            mean = zs[0]
            std = zs[1]
            res = (val - mean) / std
        return res

    def load_conf(self, conf_file):
        self.conf = {}
        f = open(conf_file)
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line[0] = '#':
                continue
            lis = line.split('=')
            if len(list) != 2:
                continue
            self.conf[lis[0]] = lis[1]
        f.close()

    def init_conf(self):
        self.r = 0.0
        if self.g("enable_sample") == '1':
            pos_rate = float(self.g("pos_rate"))
            rate = float(self.g("rate"))
            sub_rate = 0.0
            x = self.g("sub_rate")
            if x != '':
                sub_rate = float(x)
            if sub_rate > 1e-6 and sub_rate < rate:
                self.r = sub_rate / rate
            else:
                self.r = pos_rate / (1 - pos_rate) * rate
        else:
            sys.stderr.write("sample_mod config fail\n")
            exit(1)

        self.need_filt_slot = False
        self.need_normalize = False
        self.filt_slot_set = set()
        self.zs_dic = {}

        if self.g("enable_fea_filt") == '1' and self.g("fea_filt_slot_conf") != '':
            self.filt_slot_set = self.load_filtslot(self.g("fea_filt_slot_conf"))
            if len(self.filt_slot_set) > 0:
                self.need_filt_slot = True
        if self.g("zscore_file") != '':
            self.zs_dic = self.load_zscore(self.g("zscore_file"))
            if len(self.zs_dic) > 0:
                self.need_normalize = True

    def parse_line(self, line):
        line = line.split('#', 1)[0].strip()
        if line == '':
            return
        lis = line.split(' ')
        label = lis[0]
        if self.is_test == False and label == '0' and self.g("enable_sample") == '1':
            if self.neg_sample(self.r) == False:
                return
        if self.need_filt_slot == False and self.need_normalize == False:
            print(line)
            return
        out_lis = []
        out_lis.append(label)
        for i in xrange(1, len(lis)):
            li = lis[i].split(':')
            if len(li) != 2:
                out_lis.append(lis[i])
                continue
            slot = int(li[0])
            value = float(li[1])
            if self.need_normalize:
                val = self.normal(slot, value, self.zs_dic)
            if self.need_filt_slot:
                if slot in self.filt_slot_set:
                    val = 0.0
                    continue
            out_lis.append(str(slot)+':'+str(val))
        print(' '.join(out_lis))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("input at least conf_file")
        exit(1)
    conf_file = sys.argv[1]

    is_test = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "test":
            is_test = True
    pre = PreProcess(conf_file=conf_file, is_test=is_test)
    for line in sys.stdin:
        pre.parse_line(line)
