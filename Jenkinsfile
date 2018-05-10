pipeline {
    agent {
	docker { image 'z317999579/t2t:157' }
    }
    stages {
	stage('Test') {
            steps {
                sh '''
			cd predict/app/script
			python test_problemDecoderPredictMode_transformerscorer.py
		'''
            }
        }
    }
    post {
        always {
            echo 'This will always run'
        }
        success {
            echo 'This will run only if successful'
        }
        failure {
            echo 'This will run only if failed'
        }
        unstable {
            echo 'This will run only if the run was marked as unstable'
        }
        changed {
            echo 'This will run only if the state of the Pipeline has changed'
            echo 'For example, if the Pipeline was previously failing but is now successful'
        }
    }
}
