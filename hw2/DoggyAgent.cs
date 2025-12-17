using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System;
using Random = UnityEngine.Random;
using UnityEngine.InputSystem;


public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    public float strenghtMove;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    private float distToTarget = 0f;

    [Header("Rewards (tuning)")]

    [Tooltip("Награда за уменьшение расстояния до куба.")]
    public float progressRewardScale = 5.0f;

    [Tooltip("Клип прогресса на шаг, чтобы не было сильных всплесков.")]
    public float progressClip = 0.25f;

    [Tooltip("Бонус за обновление лучшей дистанции в эпизоде.")]
    public float bestImproveBonus = 0.10f;

    [Tooltip("Мягкий штраф за время.")]
    public float stepPenalty = 0.0015f;

    [Tooltip("Штраф, если почти не двигается и нет прогресса.")]
    public float hardIdlePenalty = 0.02f;

    [Tooltip("Порог скорости, ниже которого считаем, что стоит.")]
    public float idleSpeedThreshold = 0.20f;

    [Tooltip("Порог прогресса, ниже которого считаем, что прогресса нет.")]
    public float idleProgressThreshold = 0.003f;

    [Tooltip("Бонус за вертикальность (устойчивость).")]
    public float uprightRewardScale = 0.01f;

    [Tooltip("Штраф за большую угловую скорость (чтобы не вертелся).")]
    public float angularVelocityPenaltyScale = 0.00015f;

    [Tooltip("Штраф за энергию действий (небольшой).")]
    public float actionL2PenaltyScale = 0.00015f;

    [Tooltip("Штраф за дерганье действий (небольшой).")]
    public float actionDeltaPenaltyScale = 0.00008f;

    [Tooltip("Сколько шагов в начале эпизода не штрафуем за дерганье.")]
    public int actionDeltaGraceSteps = 80;

    [Tooltip("Считаем, что упал, если центр тела опустился ниже этого Y.")]
    public float fallYThreshold = 0.10f;

    [Tooltip("Штраф при падении.")]
    public float fallPenalty = -1.8f;

    [Tooltip("Расстояние до куба, при котором считаем, что успех.")]
    public float successDistance = 1.0f;

    [Tooltip("Награда за достижение куба.")]
    public float successReward = 6.0f;

    [Tooltip("Если долго нет улучшения лучшего расстояния — завершаем эпизод (чтобы не стоял).")]
    public int stagnationSteps = 180;

    [Tooltip("Штраф при стоянии.")]
    public float stagnationPenalty = -1.2f;

    [Tooltip("Штраф за угол между 'вперёд' и направлением на куб (0..1).")]
    public float angleToTargetPenaltyScale = 0.01f;

    [Tooltip("Бонус за уменьшение |угла|.")]
    public float angleImproveRewardScale = 0.015f;

    [Tooltip("Штраф за боковую скорость (перпендикулярно направлению на куб).")]
    public float lateralVelocityPenaltyScale = 0.02f;

    [Tooltip("Штраф за скорость поворота вокруг вертикали.")]
    public float yawRatePenaltyScale = 0.0002f;

    [Tooltip("Штраф за наклон (в градусах) вокруг оси 'вперёд'.")]
    public float rollAnglePenaltyScale = 0.018f;

    [Tooltip("Штраф за скорость вращения вокруг оси 'вперёд'.")]
    public float rollRatePenaltyScale = 0.00035f;

    [Tooltip("Наклоны меньше этого (градусы) не штрафуем.")]
    public float rollFreeDeg = 12f;

    [Tooltip("Наклоны >= этого (градусы) считаем 100% штрафа.")]
    public float rollMaxDeg = 55f;
    // -----------------------------------------------------------------------

    private float bestDistanceThisEpisode = 0f;
    private int stepsSinceBest = 0;
    private int episodeStep = 0;
    private float[] prevActions = new float[12];

    private float prevAbsAngleToTarget = 0f;
    // ------------------------------------------------------------

    public override void Initialize()
    {
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        defRot = body.transform.rotation;
        defPos = body.transform.position;

        for (int i = 0; i < 12; i++) prevActions[i] = 0f;
        prevAbsAngleToTarget = 0f;
    }

    public void ResetDog()
    {
        Quaternion newRot = Quaternion.Euler(-90, 0, Random.Range(0f, 360f));

        body.TeleportRoot(defPos, newRot);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < 12; i++)
        {
            MoveLeg(legs[i], 0);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic");
    }

    public override void OnEpisodeBegin()
    {
        ResetDog();

        cube.transform.position = new Vector3(Random.Range(-7.5f, 7.5f), 0.21f, Random.Range(-7.5f, 7.5f));

        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        bestDistanceThisEpisode = distToTarget;
        stepsSinceBest = 0;
        episodeStep = 0;
        for (int i = 0; i < 12; i++) prevActions[i] = 0f;

        Vector3 toTarget0 = cube.transform.position - body.transform.position;
        Vector3 dir0 = toTarget0.sqrMagnitude > 1e-6f ? toTarget0.normalized : body.transform.right;
        float a0 = Mathf.Abs(Vector3.SignedAngle(body.transform.right, dir0, Vector3.up));
        prevAbsAngleToTarget = a0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(body.transform.position);
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);
        sensor.AddObservation(body.transform.right);

        sensor.AddObservation(cube.transform.position);

        Vector3 relativePosition = cube.transform.position - body.transform.position;
        sensor.AddObservation(relativePosition);

        Vector3 toCube = (cube.transform.position - body.transform.position).normalized;
        float angleToCube = Vector3.SignedAngle(body.transform.right, toCube, Vector3.up);
        sensor.AddObservation(angleToCube);

        float distanceToCube = Vector3.Distance(body.transform.position, cube.transform.position);
        sensor.AddObservation(distanceToCube);

        foreach (var leg in legs)
        {
            sensor.AddObservation(leg.xDrive.target);
            sensor.AddObservation(leg.velocity);
            sensor.AddObservation(leg.angularVelocity);
        }

        foreach (var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround);
        }
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        Debug.Log("RL step");
        var actions = vectorAction.ContinuousActions;

        for (int i = 0; i < 12; i++)
        {
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, (actions[i] + 1) * 0.5f);
            MoveLeg(legs[i], angle);
        }

        episodeStep++;

        Vector3 toTarget = cube.transform.position - body.transform.position;
        float currentDistanceToTarget = toTarget.magnitude;

        if (body.transform.position.y < fallYThreshold)
        {
            AddReward(fallPenalty);
            EndEpisode();
            return;
        }

        if (currentDistanceToTarget <= successDistance)
        {
            AddReward(successReward);
            EndEpisode();
            return;
        }

        if (currentDistanceToTarget < bestDistanceThisEpisode - 0.03f)
        {
            bestDistanceThisEpisode = currentDistanceToTarget;
            stepsSinceBest = 0;
            AddReward(bestImproveBonus);
        }
        else
        {
            stepsSinceBest++;
            if (stepsSinceBest > stagnationSteps)
            {
                AddReward(stagnationPenalty);
                EndEpisode();
                return;
            }
        }
        

        // ---------- Rewards ----------
        float progress = distToTarget - currentDistanceToTarget;
        progress = Mathf.Clamp(progress, -progressClip, progressClip);
        AddReward(progress * progressRewardScale);

        float upDot = Mathf.Clamp01(Vector3.Dot(body.transform.up, Vector3.up)); 
        float upright = Mathf.Clamp((upDot - 0.7f) / 0.3f, -1f, 1f);
        AddReward(upright * uprightRewardScale);

        AddReward(-body.angularVelocity.magnitude * angularVelocityPenaltyScale);

        AddReward(-stepPenalty);

        if (body.velocity.magnitude < idleSpeedThreshold && Mathf.Abs(progress) < idleProgressThreshold)
        {
            AddReward(-hardIdlePenalty);
        }

        if (currentDistanceToTarget > 1e-6f)
        {
            Vector3 dirToTarget = toTarget / currentDistanceToTarget;

            float signedAngle = Vector3.SignedAngle(body.transform.right, dirToTarget, Vector3.up);
            float absAngle = Mathf.Abs(signedAngle);  
            float absAngle01 = absAngle / 180f;       

            AddReward(-absAngle01 * angleToTargetPenaltyScale);

            float improve = (prevAbsAngleToTarget - absAngle) / 180f; 
            AddReward(improve * angleImproveRewardScale);
            prevAbsAngleToTarget = absAngle;

            float vAlong = Vector3.Dot(body.velocity, dirToTarget);
            Vector3 vPerp = body.velocity - vAlong * dirToTarget;
            AddReward(-vPerp.magnitude * lateralVelocityPenaltyScale);

            float yawRate = Mathf.Abs(Vector3.Dot(body.angularVelocity, Vector3.up));
            AddReward(-yawRate * yawRatePenaltyScale);

            Vector3 forwardAxis = body.transform.right;

            float rollRate = Mathf.Abs(Vector3.Dot(body.angularVelocity, forwardAxis));
            AddReward(-rollRate * rollRatePenaltyScale);

            Vector3 upProj = Vector3.ProjectOnPlane(body.transform.up, forwardAxis);
            Vector3 worldUpProj = Vector3.ProjectOnPlane(Vector3.up, forwardAxis);

            if (upProj.sqrMagnitude > 1e-6f && worldUpProj.sqrMagnitude > 1e-6f)
            {
                upProj.Normalize();
                worldUpProj.Normalize();

                float rollSignedDeg = Vector3.SignedAngle(worldUpProj, upProj, forwardAxis);
                float rollAbsDeg = Mathf.Abs(rollSignedDeg);

                float t = Mathf.InverseLerp(rollFreeDeg, rollMaxDeg, rollAbsDeg); 
                t = Mathf.Clamp01(t);

                AddReward(-t * rollAnglePenaltyScale);
            }
        }

        float l2 = 0f;
        float delta = 0f;
        for (int i = 0; i < 12; i++)
        {
            float a = Mathf.Clamp(actions[i], -1f, 1f);
            l2 += a * a;

            float d = a - prevActions[i];
            delta += d * d;

            prevActions[i] = a;
        }
        l2 /= 12f;
        delta /= 12f;

        AddReward(-l2 * actionL2PenaltyScale);
        if (episodeStep > actionDeltaGraceSteps)
        {
            AddReward(-delta * actionDeltaPenaltyScale);
        }

        distToTarget = currentDistanceToTarget;
    }

    public void FixedUpdate()
    {
        body.AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove);
        for (int i = 0; i < 12; i++)
        {
            legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 20f);
        }

        RaycastHit hit;
        if (Physics.Raycast(body.transform.position, body.transform.right, out hit))
        {
            if (hit.collider.gameObject == cube)
            {
                body.AddForce(2f * strenghtMove * (cube.transform.position - body.transform.position).normalized);
                for (int i = 0; i < 12; i++)
                {
                    legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 10f);
                }
            }
        }
        Debug.DrawRay(body.transform.position, body.transform.right, Color.white);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
}
