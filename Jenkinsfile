pipeline {
    agent {
        docker { image 'rxthinking:tensor2tensor_1.5.7' }
    }
    stages {
	stage('Test') {
            steps {
                sh '''
			cd predict/app/script
			python ProblemDecoder_predict.py
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
