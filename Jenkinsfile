pipeline {
  agent {
    node { label "docker-host" }
  }

  environment {
    GIT_NAME  = "rag-facts-check"
    registry  = "eeacms/rag-facts-check"
    // Rancher catalog path - set once a catalog entry exists for this image.
    template  = ""
    DEPENDENT_DOCKERFILE_URL = ""
    dockerImage = ''
    tagName     = ''
  }

  stages {

    stage('Lint & Test') {
      when {
        allOf {
          not { buildingTag() }
          environment name: 'CHANGE_ID', value: ''
        }
      }
      steps {
        script {
          // The EEA Jenkins docker daemon is remote, so `docker run -v $PWD`
          // bind mounts are not visible to it. Bake code + tests into an
          // image (Dockerfile `test` stage) and copy the junit report out.
          def img = "$registry:ci-${env.BUILD_NUMBER}"
          def container = "${GIT_NAME}-ci-${env.BUILD_NUMBER}"
          try {
            sh "docker build --no-cache --target test -t ${img} ."

            // Lint / format: report but do not fail the build.
            catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
              sh "docker run --rm --name='${container}-lint' ${img} sh -c 'ruff check rag_facts_check/ tests/ scripts/ && ruff format --check rag_facts_check/ tests/ scripts/'"
            }

            // Tests: hard failure, but always pull the junit report out first.
            def rc = sh(returnStatus: true, script: "docker run --name='${container}' ${img} pytest --junitxml=/app/junit.xml")
            sh "docker cp '${container}:/app/junit.xml' junit.xml || true"
            if (rc != 0) {
              error("pytest failed (exit ${rc})")
            }
          } finally {
            sh "docker rm -f '${container}' || true"
            sh "docker rmi ${img} || true"
          }
        }
      }
      post {
        always {
          junit testResults: 'junit.xml', allowEmptyResults: true
        }
      }
    }

    stage('Docker build & push ( on branch )') {
      when {
        allOf {
          not { buildingTag() }
          environment name: 'CHANGE_ID', value: ''
        }
      }
      steps {
        script {
          if (env.BRANCH_NAME == 'main' || env.BRANCH_NAME == 'master') {
            tagName = 'latest'
          } else {
            tagName = "$BRANCH_NAME"
          }
          def date = sh(returnStdout: true, script: 'echo $(date "+%Y-%m-%dT%H%M")').trim()
          try {
            dockerImage = docker.build("$registry:$tagName", "--no-cache .")
            docker.withRegistry('', 'eeajenkins') {
              dockerImage.push()
              dockerImage.push(date)
            }
          } finally {
            sh "docker rmi $registry:$tagName || true"
          }
        }
      }
    }

    stage('Release ( on tag )') {
      when {
        buildingTag()
      }
      steps {
        node(label: 'docker') {
          withCredentials([string(credentialsId: 'eea-jenkins-token', variable: 'GITHUB_TOKEN'), usernamePassword(credentialsId: 'jekinsdockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
            sh '''docker pull eeacms/gitflow; docker run -i --rm --name="$BUILD_TAG-release" \
              -e GIT_BRANCH="$BRANCH_NAME" \
              -e GIT_NAME="$GIT_NAME" \
              -e DOCKERHUB_REPO="$registry" \
              -e GIT_TOKEN="$GITHUB_TOKEN" \
              -e DOCKERHUB_USER="$DOCKERHUB_USER" \
              -e DOCKERHUB_PASS="$DOCKERHUB_PASS" \
              -e DEPENDENT_DOCKERFILE_URL="$DEPENDENT_DOCKERFILE_URL" \
              -e RANCHER_CATALOG_PATHS="$template" \
              -e GITFLOW_BEHAVIOR="RUN_ON_TAG" \
              eeacms/gitflow'''
          }
        }
      }
    }

  }

  post {
    always {
      cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, deleteDirs: true)
    }
    changed {
      script {
        def url = "${env.BUILD_URL}/display/redirect"
        def status = currentBuild.currentResult
        def details = """<h1>${env.JOB_NAME} - Build #${env.BUILD_NUMBER} - ${status}</h1>
                         <p>Check console output at <a href="${url}">${env.JOB_BASE_NAME} - #${env.BUILD_NUMBER}</a></p>
                      """
        emailext(
          subject: '$DEFAULT_SUBJECT',
          body: details,
          attachLog: true,
          compressLog: true,
          recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'CulpritsRecipientProvider']]
        )
      }
    }
  }
}
