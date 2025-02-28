pipeline {
    agent any

    environment {
        CLEAN_IMAGE = 'mlops:latest'
        ARTIFACTS_DIR = 'model_artifacts'
    }

    stages {
        
        stage('creation repertoire travail') {
            steps {
                bat "if not exist ${ARTIFACTS_DIR} mkdir ${ARTIFACTS_DIR}"

            }
        }
        
        stage('Build Docker Clean-Data') {
            steps {
                script {
                    bat 'docker build -t %CLEAN_IMAGE% .'
                }
            }
        }

        stage('Nettoyer les données et entrainement') {
            steps {
                script {
                    bat "docker run --rm -v ${WORKSPACE}/${ARTIFACTS_DIR}:/work/artefact ${CLEAN_IMAGE}"
                }
            }
        }

        stage('Démarrer les services') {
            steps {
                bat 'docker-compose up -d'
            }
        }

        stage('Sauvegarder les artefacts') {
             steps {
                script {
                    // S'assurer que les artefacts sont bien stockés dans le répertoire local
                    bat "if exist ${ARTIFACTS_DIR} dir ${ARTIFACTS_DIR} /s"
                }
                archiveArtifacts artifacts: "**/*.pkl", fingerprint: true
                archiveArtifacts artifacts: "**/*.json", fingerprint: true
            }
        }

    }

    post {
        success {
            echo "Pipeline terminé avec succès"
        }
        failure {
            echo "Échec du pipeline"
        }
    }
}
