pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    timeout(time: 2, unit: 'HOURS')
  }
  stages {
    stage('test') {
      parallel {
        stage('linux-python2') {
          agent {
            dockerfile {
              dir "test/linux-python2"
              args '-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v /home/jenkins/.conda2/pkgs:/home/jenkins/.conda/pkgs:rw,z'
            }
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}/test/${env.STAGE_NAME}"
          }
          steps {
            sh 'conda env create -q -f environment_python2.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $CONDA_ENV/bin/activate $CONDA_ENV
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
              cd $TEMPDIR
              caimanmanager.py install
              nosetests --traverse-namespace caiman
              caimanmanager.py demotest
            '''
          }
        }
        stage('linux-python3') {
          agent {
            dockerfile {
              dir "test/linux-python3"
              args '-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v /home/jenkins/.conda3/pkgs:/home/jenkins/.conda/pkgs:rw,z'
            }
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}/test/${env.STAGE_NAME}"
          }
          steps {
            sh 'conda env create -q -f environment.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $CONDA_ENV/bin/activate $CONDA_ENV
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
              cd $TEMPDIR
              caimanmanager.py install
              nosetests --traverse-namespace caiman
              caimanmanager.py demotest
            '''
          }
        }

        stage('osx-python2') {
          agent {
            label 'osx && anaconda2'
          }
          environment {
            CONDA_ENV = "${env.WORKSPACE}/test/${env.STAGE_NAME}"
          }
          steps {
            sh '$ANACONDA2/bin/conda env create -q -f environment_python2.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $CONDA_ENV/bin/activate $CONDA_ENV
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
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
              source $CONDA_ENV/bin/activate $CONDA_ENV
              pip install .
              TEMPDIR=$(mktemp -d)
              export CAIMAN_DATA=$TEMPDIR/caiman_data
              cd $TEMPDIR
              caimanmanager.py install
              nosetests --traverse-namespace caiman
              caimanmanager.py demotest
            '''
          }
        }

        /*
        stage('win-python2') {
          agent {
            label 'windows && anaconda2'
          }
          environment {
            ANACONDA = "C:\\ProgramData\\Anaconda2"
            CONDA_ENV = "${env.WORKSPACE}\\test\\${env.STAGE_NAME}"
          }
          steps {
            bat '%ANACONDA%\\scripts\\conda env create -q -f environment_python27.yml -p %CONDA_ENV%'
            bat '%CONDA_ENV%\\scripts\\activate %CONDA_ENV% && pip install . && caimanmanager.py install && nosetests'
          }
        }
        */
        stage('win-python3') {
          agent {
            label 'windows && anaconda3'
          }
          environment {
            ANACONDA = "C:\\ProgramData\\Anaconda3"
            CONDA_ENV = "${env.WORKSPACE}\\test\\${env.STAGE_NAME}"
          }
          steps {
            bat '%ANACONDA%\\scripts\\conda env create -q -f environment.yml -p %CONDA_ENV%'
            bat '%CONDA_ENV%\\scripts\\activate %CONDA_ENV% && pip install . && copy caimanmanager.py %TEMP% && cd %TEMP% && set "CAIMAN_DATA=%TEMP%\\caiman_data" && (if exist caiman_data (rmdir caiman_data /s /q) else (echo "Host is fresh")) && python caimanmanager.py install && python caimanmanager.py test'
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
	       to: 'epnevmatikakis@gmail.com, andrea.giovannucci@gmail.com, dsimon@flatironinstitute.org, pgunn@flatironinstitute.org'
    }
  }
}
