import ast
import sys, traceback

sys.path.insert(0, 'utils')
from code_utils import toks2lines


def check_paren_error(code_toks_raw):
    #
    parenlev = [0,0,0]
    paren_open_tidx = [None, None, None]
    tidx = 0
    tidx2linenocol = [] #lineno is 1-start. col is 0-start
    code_lines = toks2lines(code_toks_raw)
    for lidx, line in enumerate(code_lines):
        for col, tok in enumerate(line):
            tidx2linenocol.append([lidx+1, col])
            if tok == '(':
                if parenlev[0] == 0:
                    paren_open_tidx[0] = tidx
                parenlev[0] += 1
            elif tok == '{':
                if parenlev[1] == 0:
                    paren_open_tidx[1] = tidx
                parenlev[1] += 1
            elif tok == '[':
                if parenlev[2] == 0:
                    paren_open_tidx[2] = tidx
                parenlev[2] += 1
            elif tok == ')':
                parenlev[0] -= 1
                if parenlev[0] == 0:
                    paren_open_tidx[0] = None
            elif tok == '}':
                parenlev[1] -= 1
                if parenlev[1] == 0:
                    paren_open_tidx[1] = None
            elif tok == ']':
                parenlev[2] -= 1
                if parenlev[2] == 0:
                    paren_open_tidx[2] = None
            if parenlev[0] < 0 or parenlev[1] < 0 or parenlev[2] < 0:
                err_obj = {
                            'msg': 'unbalanced (){}[]',
                            'msg_detailed': 'extra right parenthesis',
                          }
                return err_obj
            tidx += 1
    if parenlev != [0,0,0]:
        assert (paren_open_tidx[0]!=None) or (paren_open_tidx[1]!=None) or (paren_open_tidx[2]!=None)
        paren_open_tidx_earliest = min([_ for _ in paren_open_tidx if _ is not None])
        err_obj = {
                    'msg': 'unbalanced (){}[]',
                    'msg_detailed': 'left parenthesis is not closed',
                  }
        return err_obj
    #No error
    return 0

def check_ast_error(codeString):
    try:
        tree = ast.parse(codeString, filename='NA')
    except SyntaxError:
        value = sys.exc_info()[1]
        tb_string = traceback.format_exc()
        msg = value.args[0]
        (lineno, offset, text) = value.lineno, value.offset, value.text
        if text is None:
            lines = codeString.splitlines()
            if len(lines) >= lineno:
                text = lines[lineno - 1]
                if sys.version_info >= (3, ) and isinstance(text, bytes):
                    try:
                        text = text.decode('ascii')
                    except UnicodeDecodeError:
                        text = None
        offset -= 1
        if text is None:
            return 1 #Exception
        else:
            line = text.splitlines()[-1]
            if offset is not None:
                if sys.version_info < (3, 8):
                    offset = offset - (len(text) - len(line)) + 1
            else:
                offset = 0
            err_obj = {'lineno': lineno, #starts from 1
                        'col': offset,   #starts from 0
                        'line_content': line,
                        'msg': msg,
                      }
            return err_obj
    except Exception:
        return 1 #Exception
    #no ast error
    return 0
