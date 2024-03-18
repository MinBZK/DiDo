"""
This module contains S3 helper functions
"""
import os
import subprocess



def s3_clean_msg(clean_me:str, split_on:str='s3://', starts_with:str='s3_dg', \
    remove_me:str='') -> list:
    """use within helper for s3cmd messages

    Args:
        clean_me -- s3cmd message
        split_on -- split string on identifier
        starts_with -- each returnable list item
        remove_me -- matching items from return list (sometimes also returns main filepath)

    Returns:
        list with clean message

    Example:
    In:
        CompletedProcess(
        args=['s3cmd', 'ls', 's3://s3_dgdoobi_dwh_migratie'], returncode=0, stdout=b'
        DIR  s3://s3_dgdoobi_dwh_migratie/DW-DBS001/\n
        DIR  s3://s3_dgdoobi_dwh_migratie/DW-DBS002/\n
        DIR  s3://s3_dgdoobi_dwh_migratie/_MAPPEN/\n'
        stderr=b'')
    Out:
        [s3://s3_dgdoobi_dwh_migratie/DW-DBS001/\n,
         s3://s3_dgdoobi_dwh_migratie/DW-DBS002/\n,
         s3://s3_dgdoobi_dwh_migratie/_MAPPEN/\n]
    """
    clean_me = clean_me.split(split_on)
    clean_me = [i.split('\\n') for i in clean_me]
    clean_me = [i for sublist in clean_me for i in sublist]
    clean_me = [i for i in clean_me if i.startswith(starts_with)]
    clean_me = [split_on+i for i in clean_me]
    clean_me = [i for i in clean_me if not i == remove_me]

    return clean_me


def s3_command_ls_return_fullpath(folder:str='', bucket:str='prd') -> list:
    """
        listing without details on files or folders
        raises error if S3 credentials are incomplete

    Args:
        folder -- folderpath
        bucket -- S3 bucket

    Returns:
        list of all items found in the specific folder
    """
    # errcheck
    if folder != '':
        if not folder.startswith('/'):
            folder = '/' + folder
        if not folder.endswith('/'):
            folder += '/'

    # run process
    cmd = ['s3cmd', 'ls', f's3://s3_dgdoobi_dwh_{bucket}{folder}']
    executed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # check for runtime errors
    out = str(executed.stderr)
    if 'ERROR:' in out.upper():
        out = s3_clean_msg(clean_me=out, split_on='ERROR', starts_with=':')
        raise RuntimeError(('\n').join(out))

    out = str(executed.stdout)
    out = s3_clean_msg(clean_me=out, split_on='s3://', starts_with='s3_dg', remove_me=cmd[-1])

    return out


def s3_command_get_file(download_to:str, filepath_s3:str, force_overwrite:bool=False) -> str:
    """
        downloads file to local environment
        ideal to combine with s3_command_ls and iterative over objects

    Args:
        download_to -- full filepath to local env
        filepath_s3 -- full filepath ('s3://s3_dgdoobi_dwh_bucket/folder/file.ext')
        force_overwrite -- True overwrites existing local file

    Returns:
        printed 'download from s3' statement
    """
    # input errcheck
    # if not os.path.exists(download_to):
    #     raise FileNotFoundError(download_to)

    # run process
    cmd = ['s3cmd', '-f', 'get', filepath_s3, download_to]
    if not force_overwrite:
        cmd.pop(1)

    executed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # check for runtime errors
    out = str(executed.stderr)
    if 'ERROR:' in out.upper():
        out = s3_clean_msg(clean_me=out, split_on='ERROR', starts_with=':')
        raise RuntimeError(('\n').join(out))

    out = str(executed.stdout)
    out = s3_clean_msg(clean_me=out, split_on='download:', starts_with=" '")
    print(''.join(out))

    return out


def s3_command_put_file(filepath_local:str, filepath_s3:str, force_overwrite:bool=False) -> str:
    """uploads file from local environment to S3, makes subfolders if not exist on S3

    Args:
        filepath_local -- full filepath file to be uploaded
        filepath_s3 -- full s3 folderpath ('s3://s3_dgdoobi_dwh_prd/folder/')
        force_overwrite -- True overwrites existing S3 file

    Raises:
        if file already exists on S3 (unless force_overwrite==True)

    Returns:
        filepath s3 of uploaded file
    """
    # input errcheck
    if not os.path.exists(filepath_local):
        raise FileNotFoundError(filepath_local)
    if not filepath_s3.endswith('/'):
        filepath_s3 += '/'

    if not force_overwrite:
        # check if file exists on s3, removing s3://s3_dgdoobi_dwh_prd/ for ls function
        files_on_s3 = s3_command_ls_return_fullpath(folder=filepath_s3.split('/', maxsplit=3)[-1])
        files_on_s3 = [os.path.basename(i) for i in files_on_s3]
        files_on_s3 = [i for i in files_on_s3 if os.path.basename(filepath_local) in i]

        if len(files_on_s3) > 0:
            raise FileExistsError(filepath_s3+files_on_s3[0])

    # run process
    cmd = ['s3cmd', 'put', filepath_local, filepath_s3]
    executed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # check for runtime errors
    out = str(executed.stderr)
    if 'ERROR:' in out.upper():
        out = s3_clean_msg(clean_me=out, split_on='ERROR', starts_with=':')
        raise RuntimeError(('\n').join(out))

    return str(executed.stdout).split()[3]
