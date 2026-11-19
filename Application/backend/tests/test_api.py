"""End-to-end API test with a throwaway database and real OLID-I leaf photos.

Run from the repository root:  python -m pytest Application/backend/tests -q
"""
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.environ['PLANTCARE_DATA_DIR'] = tempfile.mkdtemp(prefix='plantcare_test_')
sys.path.insert(0, os.path.join(ROOT, 'Application', 'backend'))

from fastapi.testclient import TestClient  # noqa: E402

from app.conditions import CROPS  # noqa: E402
from app.main import app, predictor  # noqa: E402

PHOTOS = os.path.join(ROOT, 'Research_Paper', 'data', 'illustrative_images')


def leaf_photo(filename):
    with open(os.path.join(PHOTOS, filename), 'rb') as f:
        return f.read()


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def signup(client, email):
    r = client.post('/api/auth/register', json={'name': 'Test User', 'email': email, 'password': 'secret123'})
    assert r.status_code == 201, r.text
    return {'Authorization': f"Bearer {r.json()['token']}"}


def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['models_loaded'] is True


def test_auth_flow(client):
    h = signup(client, 'alice@example.com')
    assert client.get('/api/auth/me', headers=h).json()['email'] == 'alice@example.com'
    assert client.post('/api/auth/register', json={'name': 'A', 'email': 'ALICE@example.com', 'password': 'secret123'}).status_code == 409
    assert client.post('/api/auth/login', json={'email': 'alice@example.com', 'password': 'wrong'}).status_code == 401
    assert client.post('/api/auth/login', json={'email': 'Alice@Example.com', 'password': 'secret123'}).status_code == 200
    assert client.get('/api/predictions').status_code == 401
    assert client.get('/api/predictions', headers={'Authorization': 'Bearer junk'}).status_code == 401


def test_diagnosis_and_history(client):
    h = signup(client, 'bob@example.com')
    r = client.post('/api/predict', content=leaf_photo('20_STRESSED_tomato_LM.JPG'),
                    headers={**h, 'Content-Type': 'image/jpeg', 'X-Filename': 'tomato%20leaf.jpg'})
    assert r.status_code == 201, r.text
    rec = r.json()
    res = rec['result']
    assert rec['filename'] == 'tomato leaf.jpg'
    assert res['crop']['code'] in CROPS and res['crop']['source'] == 'model' and len(res['conditions']) == 3
    assert abs(sum(a['probability'] for a in res['crop']['alternatives']) - 1) < 0.5  # top-3 of a distribution
    assert res['stress']['status'] in ('healthy', 'stressed') and len(res['explanation']) == 5
    assert client.get(rec['image_url']).status_code == 200 and client.get(rec['overlay_url']).status_code == 200

    # known-crop override
    r2 = client.post('/api/predict?crop=ash_gourd', content=leaf_photo('01_NON-STRESSED_ash_gourd_healthy.JPG'),
                     headers={**h, 'Content-Type': 'image/jpeg'})
    assert r2.status_code == 201 and r2.json()['result']['crop']['code'] == 'ash_gourd'
    assert r2.json()['result']['crop']['source'] == 'user'

    items = client.get('/api/predictions', headers=h).json()
    assert len(items) == 2
    assert client.get('/api/predictions/stats', headers=h).json()['total'] == 2
    assert client.patch(f"/api/predictions/{rec['id']}", json={'note': 'field 3'}, headers=h).json()['note'] == 'field 3'

    # another user cannot see or delete it
    other = signup(client, 'carol@example.com')
    assert client.get(f"/api/predictions/{rec['id']}", headers=other).status_code == 404
    assert client.delete(f"/api/predictions/{rec['id']}", headers=other).status_code == 404

    assert client.delete(f"/api/predictions/{rec['id']}", headers=h).status_code == 204
    assert client.get(rec['image_url']).status_code == 404
    assert len(client.get('/api/predictions', headers=h).json()) == 1


def test_bad_uploads(client):
    h = signup(client, 'dave@example.com')
    assert client.post('/api/predict', content=b'not an image', headers=h).status_code == 422
    assert client.post('/api/predict', content=b'', headers=h).status_code == 400
    assert client.post('/api/predict?crop=banana', content=leaf_photo('07_NON-STRESSED_tomato_healthy.JPG'), headers=h).status_code == 422


def test_stress_verdict_uses_evaluated_svm_rule(client, monkeypatch):
    model = predictor().crop_models['tomato']['stress']
    svc = model.svc_
    # Force disagreement regardless of the library's fitted calibration.
    def opposite_probability(kernel):
        return np.array([[0.8, 0.2] if label == 1 else [0.2, 0.8] for label in svc.predict(kernel)])

    monkeypatch.setattr(svc, 'predict_proba', opposite_probability)
    headers = signup(client, 'boundary@example.com')
    response = client.post('/api/predict?crop=tomato', headers=headers, content=leaf_photo('07_NON-STRESSED_tomato_healthy.JPG'))
    assert response.status_code == 201, response.text
    result = response.json()['result']
    stress = result['stress']
    assert stress['decision_rule'] == 'svm_predict'
    assert (stress['status'] == 'stressed') == (result['trace']['svm']['decision'] > 0)
    assert stress['confidence'] == 0.2
    assert (stress['status'] == 'stressed') != (stress['probability_stressed'] >= 0.5)
