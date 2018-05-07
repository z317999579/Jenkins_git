pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh 'javac hello.java'
		sh 'java hello'
            }
        }
    }
}
