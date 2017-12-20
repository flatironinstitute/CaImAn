pipeline {
  agent none
  options {
    timeout(time: 1, unit: 'HOURS')
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
              python setup.py build_ext -i
              nosetests
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
              python setup.py build_ext -i
              nosetests
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
            sh '$ANACONDA2/bin/conda env create -q -f environment_mac_python2.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $CONDA_ENV/bin/activate $CONDA_ENV
              python setup.py build_ext -i
              #nosetests
              cd caiman/tests
              nosetests $(for f in test_*.py ; do echo ${f%.py} ; done)
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
            sh '$ANACONDA3/bin/conda env create -q -f environment_mac.yml -p $CONDA_ENV'
            sh '''#!/bin/bash -ex
              source $CONDA_ENV/bin/activate $CONDA_ENV
              python setup.py build_ext -i
              #nosetests
              cd caiman/tests
              nosetests $(for f in test_*.py ; do echo ${f%.py} ; done)
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
            bat '%ANACONDA%\\scripts\\conda env create -q -f environment_mac_python27.yml -p %CONDA_ENV%'
            bat '%CONDA_ENV%\\scripts\\activate %CONDA_ENV% && python setup.py build_ext -i && nosetests'
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
            bat '%CONDA_ENV%\\scripts\\activate %CONDA_ENV% && python setup.py build_ext -i && nosetests'
          }
        }
      }
    }
  }
  post {
    failure {
      emailext subject: '$DEFAULT_SUBJECT',
	       body: '$DEFAULT_CONTENT',
	       recipientProviders: [
		 [$class: 'DevelopersRecipientProvider'],
	       ], 
	       replyTo: '$DEFAULT_REPLYTO',
	       to: 'epnevmatikakis@gmail.com, andrea.giovannucci@gmail.com'
    }
  }
}
