pipeline {
    agent any
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
	    mail to:'joy317999579@gmail.com',
		subject:'successful pipeline: test pipeline',
		body:'successfully run pipeline'
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
