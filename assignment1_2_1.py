from sklearn.svm import LinearSVC


def simple_svc_linear_decision_boundry():
    
    model = LinearSVC(dual='auto',verbose=0)

    return model



