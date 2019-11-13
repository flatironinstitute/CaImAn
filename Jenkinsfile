pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '10', daysToKeepStr: '15'))
    timeout(time: 1, unit: 'HOURS')
    retry(3)
    timestamps()
  }
  stages {
    stage('test') {
      parallel {
        stage('linux-python3') {
          agent {
            dockerfile {
              dir "test/linux-python3"
              args '-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group'
            }
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}/test/${env.STAGE_NAME}"
            HOME = pwd(tmp:true)
          }
          steps {
            sh 'chown 993:989 /opt/conda/pkgs/urls.txt'
            sh 'conda clean --index-cache'
            sh 'conda env create -q -f environment.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source activate $CONDA_ENV
              export KERAS_BACKEND=tensorflow
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
              export THEANO_FLAGS="base_compiledir=$TEMPDIR/theano_tmp"
              cd $TEMPDIR
              caimanmanager.py install
              nosetests --traverse-namespace caiman
              caimanmanager.py demotest
            '''
          }
        }

        stage('osx-python3') {
          agent {
            label 'osx && anaconda3'
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}/test/${env.STAGE_NAME}"
            LANG = "en_US.UTF-8"
          }
          steps {
            sh '$ANACONDA3/bin/conda env create -q -f environment.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $ANACONDA3/bin/activate $CONDA_ENV
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
              cd $TEMPDIR
              caimanmanager.py install
              nosetests --traverse-namespace caiman
            '''
          }
        }

	// With the CONDA_ENV variable on windows, you must be careful not to hit the maximum path length. 
        stage('win-python3') {
          agent {
            label 'windows && anaconda3'
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}\\conda-envinst"
          }
          steps {
            bat '%ANACONDA3%\\scripts\\conda info'
            bat 'if exist "%CONDA_ENV%" rd /s /q %CONDA_ENV%'
            bat '%ANACONDA3%\\scripts\\conda env create -q --force -f environment.yml -p %CONDA_ENV%'
            bat 'if exist "%CONDA_ENV%\\etc\\conda\\activate.d\\vs*_compiler_vars.bat" del "%CONDA_ENV%\\etc\\conda\\activate.d\\vs*_compiler_vars.bat"'
            bat '%ANACONDA3%\\scripts\\activate %CONDA_ENV% && %ANACONDA3%\\scripts\\conda list'
            bat '%ANACONDA3%\\scripts\\activate %CONDA_ENV% && set KERAS_BACKEND=tensorflow && pip install . && copy caimanmanager.py %TEMP% && cd %TEMP% && set "CAIMAN_DATA=%TEMP%\\caiman_data" && (if exist caiman_data (rmdir caiman_data /s /q && echo "Removed old caiman_data" ) else (echo "Host is fresh")) && python caimanmanager.py install --force && python caimanmanager.py test'
          }
        }
      }
    }
  }
  post {
    failure {
      emailext subject: '$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS',
	       body: '''$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS

Check console output at $BUILD_URL to view full results.

Building $BRANCH_NAME for $CAUSE
$JOB_DESCRIPTION

Chages:
$CHANGES

End of build log:
${BUILD_LOG,maxLines=60}
''',
	       recipientProviders: [
		 [$class: 'DevelopersRecipientProvider'],
	       ], 
	       replyTo: '$DEFAULT_REPLYTO',
	       to: 'epnevmatikakis@gmail.com, andrea.giovannucci@gmail.com, pgunn@flatironinstitute.org'
    }
  }
}
