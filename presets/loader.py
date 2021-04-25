import os
import yaml
from munch import Munch, munchify
import collections.abc

class Preset(Munch):
    def __dict_merge(self, dct, merge_dct, prefix='', custom=False):
        for k, v in merge_dct.items():
            newprefix = k if prefix == '' else prefix + '.' + k
            if (k in dct and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], collections.abc.Mapping)):
                self.__dict_merge(dct[k], merge_dct[k], newprefix, custom=custom)
            else:
                if custom and k not in dct:
                    print('warning: unregistered parameter `{0}` with value `{1}`'.format(newprefix, merge_dct[k]))
                dct[k] = merge_dct[k]
                if custom:
                    print('parameter overriding: set `{0}` to `{1}`'.format(newprefix, merge_dct[k]))

    def __dict_append_prefix(self, dct, prefix):
        segs = prefix.split('.')
        for i in reversed(range(len(segs))):
            dct = munchify({segs[i]: dct})
        return dct

    def __load_preset(self, path, prefix):
        print('loading preset: (%s)' % prefix, path)
        with open(path, 'r') as file:
            presetdata = munchify(yaml.safe_load(file))
        if presetdata is not None:
            presetdata = self.__dict_append_prefix(presetdata, prefix)
            self.__dict_merge(self, presetdata)
    
    def __load_directory(self, path, dirname, prefix):
        for x in os.listdir(path):
            newpath = os.path.join(path, x)
            if os.path.isdir(newpath):
                newprefix = prefix + '.' + x if prefix != '' else x
                self.__load_directory(newpath, x, newprefix)
            elif os.path.isfile(newpath):
                filename = os.path.splitext(x)[0]
                newprefix = prefix
                if dirname != filename:
                    newprefix = prefix + '.' + filename if prefix != '' else filename
                self.__load_preset(newpath, newprefix)

    def load_default(self):
        dir = os.path.dirname(__file__)
        default_dir = os.path.join(dir, 'default')
        self.__load_directory(default_dir, 'default', '')
        
    def load_env_override(self):
        dir = os.path.dirname(__file__)
        override_dir = os.path.join(dir, 'env_override') 
        if os.path.isdir(override_dir):
            override_file = os.path.join(override_dir, self.experiment.env + '.yaml')
            if os.path.isfile(override_file):
                print('loading env overriding presets')
                print('-------------------------------------')
                self.load_override(override_file, custom=True)
                print('-------------------------------------')
    
    def load_override(self, path, custom=False):
        with open(path, 'r') as file:
            items = yaml.safe_load(file)
            if items is not None:
                for k,v in items.items():
                    self.__dict_merge(self, self.__dict_append_prefix(v, k), custom=custom)

    def load_custom(self, *preset_names):
        dir = os.path.dirname(__file__)
        for preset in preset_names:
            path = os.path.join(dir, 'custom', preset + '.yaml')
            print('loading custom preset:', path)
            print('-------------------------------------')
            self.load_override(path, custom=True)
            print('-------------------------------------')

    def override(self, other):
        self.__dict_merge(self, other)
