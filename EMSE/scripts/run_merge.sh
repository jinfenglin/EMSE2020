echo 'All projects'
python merge_SE_projects.py \
--root_dir G:/Document/InterMingualTraceGitData/git_projects \
--out_dir G:/Document/EMSE/all \
--allowed_projects arthas bk-cmdb canal druid Emmagee nacos ncnn pegasus QMUI_Android QMUI_iOS rax san weui xLua open-korean-text Cica awesome-berlin \
--overwrite

echo 'Chinese project only'
python merge_SE_projects.py \
--root_dir G:/Document/InterMingualTraceGitData/git_projects \
--out_dir G:/Document/EMSE/chinese_only \
--allowed_projects arthas bk-cmdb canal druid Emmagee nacos ncnn pegasus QMUI_Android QMUI_iOS rax san weui xLua \
--overwrite
