# MIT License
#
# Copyright (c) 2019 Alex Shafer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import os
import sys
from typing import Optional, Mapping, Union, Pattern

_simple_re: Pattern = re.compile(r"(?<!\\)\$([A-Za-z0-9_]+)")
_extended_re: Pattern = re.compile(r"(?<!\\)\$\{([A-Za-z0-9_]+)((:?-)([^}]+))?\}")


class envsubst:
    strict: bool
    env: Union[Optional, Mapping]

    def __init__(self):
        self.strict = False
        self.env = None

    def sub(
        self, search_string, env: Union[Optional, Mapping] = None, strict: bool = False
    ):
        """
        Substitute environment variables in the given string, allows for passing a custom environment mapping.
        The default behavior is to check the custom environment mapping , the system environment and finally the
        specified default ("somestring" in the examples). If strict is True, the system environment will not be
        checked after the custom environment mapping but, the default will still be used (if needed).

        The following forms are supported:

        Simple variables - will use an empty string if the variable is unset
          $FOO

        Bracketed expressions
          ${FOO}
            identical to $FOO
          ${FOO:-somestring}
            uses "somestring" if $FOO is unset, or is set and empty
          ${FOO-somestring}
            uses "somestring" only if $FOO is unset
        """
        self.strict = strict
        self.env = env
        # handle simple un-bracketed env vars like $FOO
        a = _simple_re.sub(self._repl_simple_env_var, search_string)
        # handle bracketed env vars with optional default specification
        b = _extended_re.sub(self._repl_extended_env_var, a)
        return b

    def _resolve_var(self, var_name, default=None):
        try:
            index = int(var_name)
            try:
                return sys.argv[index]
            except IndexError:
                return default
        except ValueError:
            if self.env:
                if not self.strict:
                    return self.env.get(var_name, os.environ.get(var_name, default))
                else:
                    return self.env.get(var_name, default)
            return os.environ.get(var_name, default)

    def _repl_simple_env_var(self, m):
        var_name = m.group(1)
        return self._resolve_var(var_name, "")

    def _repl_extended_env_var(self, m):
        var_name = m.group(1)
        default_spec = m.group(2)
        if default_spec:
            default = m.group(4)
            default = _simple_re.sub(self._repl_simple_env_var, default)
            if m.group(3) == ":-":
                # use default if var is unset or empty
                env_var = self._resolve_var(var_name)
                if env_var:
                    return env_var
                else:
                    return default
            elif m.group(3) == "-":
                # use default if var is unset
                return self._resolve_var(var_name, default)
            else:
                raise RuntimeError("unexpected string matched regex")
        else:
            return self._resolve_var(var_name, "")


def main():
    opened = False
    x = envsubst()
    f = sys.stdin
    try:
        try:
            fn = sys.argv[1]
            if fn != "-":
                f = open(fn)
                opened = True
        except IndexError:
            pass

        data = f.read()
        try:
            data = data.decode("utf-8")
        except AttributeError:
            pass

        sys.stdout.write(x.sub(data))
    finally:
        if opened:
            f.close()


if __name__ == "__main__":
    main()
