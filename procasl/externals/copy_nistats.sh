#!/bin/sh
# Script to do a local install of nistats
rm -rf tmp nistats
mkdir -p tmp/lib/python2.7/site-packages
ln -s tmp/lib/python2.7 tmp/lib/python2.6
mkdir -p tmp/bin
export PYTHONPATH=$(pwd)/tmp/lib/python2.7/site-packages:$(pwd)/tmp/lib/python2.6/site-packages
easy_install -Zeab tmp nistats
old_pwd=$(pwd)
#cd /home/varoquau/dev/joblib/
cd tmp/nistats/
python setup.py install --prefix $old_pwd/tmp
cd $old_pwd
cp -r tmp/lib/python2.7/site-packages/nistats-*.egg/nistats .
rm -rf tmp
# Needed to rewrite the doctests
find nistats -name "*.py" | xargs sed -i.bak "s/from nistats/from procasl.externals.nistats/"
find nistats -name "*.bak" | xargs rm

# Remove the tests folders to speed-up test time for process-asl.
# nistats is already tested on its own CI infrastructure upstream.
rm -r nistats/test

chmod -x nistats/*.py
