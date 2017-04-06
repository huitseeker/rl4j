package org.deeplearning4j.malmo;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import com.microsoft.msr.malmo.AgentHost;
import com.microsoft.msr.malmo.ClientPool;
import com.microsoft.msr.malmo.MissionRecordSpec;
import com.microsoft.msr.malmo.MissionSpec;
import com.microsoft.msr.malmo.TimestampedReward;
import com.microsoft.msr.malmo.TimestampedStringVector;
import com.microsoft.msr.malmo.WorldState;

import lombok.Setter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MDP Wrapper around Malmo Java Client Library
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoEnv implements MDP<MalmoBox, Integer, DiscreteSpace> {
    // Malmo Java client library depends on native library
    static {
        String malmoHome = System.getenv("MALMO_HOME");

        if (malmoHome == null)
            throw new RuntimeException("MALMO_HOME must be set to your Malmo environement.");

        System.load(malmoHome + "/Java_Examples/libMalmoJava.jnilib");
    }

    final static private int NUM_RETRIES = 10;

    final private Logger logger;

    @Setter
    MissionSpec mission = null;

    MissionRecordSpec missionRecord = null;
    ClientPool clientPool = null;

    MalmoObservationSpace observationSpace;
    MalmoActionSpace actionSpace;
    MalmoObservationPolicy framePolicy;

    AgentHost agent_host = null;
    WorldState last_world_state;
    MalmoBox last_observation;


    @Setter
    MalmoResetHandler resetHandler = null;

    /**
     * Create a MalmoEnv using XML-file mission description.
     * Equivalent to MalmoEnv( loadMissionXML( missionFileName ), missionRecord, actionSpace, observationSpace, framePolicy, clientPool )
     * @param missionFileName Name of XML file describing mission
     * @param missionRecord Malmo record specification. Ignored if set to NULL
     * @param actionSpace Malmo action space implementation
     * @param observationSpace Malmo observation space implementation
     * @param framePolicy Malmo frame policy implementation
     * @param clientPool Malmo client pool. If set to null, class will use a single Malmo client at 127.0.0.1:10000
     */
    public MalmoEnv(String missionFileName, MissionRecordSpec missionRecord, MalmoActionSpace actionSpace,
                    MalmoObservationSpace observationSpace, MalmoObservationPolicy framePolicy, ClientPool clientPool) {
        this(loadMissionXML(missionFileName), missionRecord, actionSpace, observationSpace, framePolicy, clientPool);
    }

    /**
     * Create a MalmoEnv using a mission specification object
     * @param mission Malmo mission specification.
     * @param missionRecord Malmo record specification. Ignored if set to NULL
     * @param actionSpace Malmo action space implementation
     * @param observationSpace Malmo observation space implementation
     * @param framePolicy Malmo frame policy implementation
     * @param clientPool Malmo client pool. If set to null, class will use a single Malmo client at 127.0.0.1:10000
     */
    public MalmoEnv(MissionSpec mission, MissionRecordSpec missionRecord, MalmoActionSpace actionSpace,
                    MalmoObservationSpace observationSpace, MalmoObservationPolicy framePolicy, ClientPool clientPool) {
        this.mission = mission;
        this.missionRecord = missionRecord != null ? missionRecord : new MissionRecordSpec();
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
        this.framePolicy = framePolicy;
        this.clientPool = clientPool;

        logger = LoggerFactory.getLogger(this.getClass());
    }

    /**
     * Convenience method to load a Malmo mission specification from an XML-file
     * @param filename name of XML file
     * @return Mission specification loaded from XML-file
     */
    public static MissionSpec loadMissionXML(String filename) {
        MissionSpec mission = null;
        try {
            String xml = new String(Files.readAllBytes(Paths.get(filename)));
            mission = new MissionSpec(xml, true);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return mission;
    }

    @Override
    public MalmoObservationSpace getObservationSpace() {
        return observationSpace;
    }

    @Override
    public MalmoActionSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public MalmoBox reset() {
        close();

        if (resetHandler != null) {
            resetHandler.onReset(this);
        }

        agent_host = new AgentHost();

        int i;
        for (i = 0; i < NUM_RETRIES; ++i) {
            try {
                Thread.sleep(100 + (i > 0 ? 500 * i : 0));

                if (clientPool != null)
                    agent_host.startMission(mission, clientPool, missionRecord, 0, "rl4j_0");
                else
                    agent_host.startMission(mission, missionRecord);
            } catch (Exception e) {
                logger.warn("Error starting mission: " + e.getMessage() + " Will retry " + (NUM_RETRIES - i - 1)
                                + " more times.");
                continue;
            }
            break;
        }

        if (i == NUM_RETRIES) {
            close();
            throw new MalmoConnectionError("Unable to connect to client.");
        }

        logger.info("Waiting for the mission to start");

        do {
            last_world_state = agent_host.getWorldState();
        } while (!last_world_state.getIsMissionRunning());

        last_world_state = waitForObservations();

        last_observation = observationSpace.getObservation(last_world_state);

        return last_observation;
    }

    private WorldState waitForObservations() {
        WorldState world_state;
        TimestampedStringVector observations;

        do {
            Thread.yield();
            world_state = agent_host.getWorldState();
            observations = world_state.getObservations();
        } while (observations.isEmpty() && world_state.getIsMissionRunning());

        return world_state;
    }

    private WorldState waitForObservationsAndRewards() {
        WorldState world_state;
        WorldState original_world_state = last_world_state;

        do {
            Thread.yield();
            world_state = agent_host.peekWorldState();
        } while (world_state.getIsMissionRunning()
                        && !framePolicy.isObservationConsistant(world_state, original_world_state));

        return agent_host.getWorldState();
    }

    @Override
    public void close() {
        if (agent_host != null) {
            agent_host.delete();
        }

        agent_host = null;
    }

    @Override
    public StepReply<MalmoBox> step(Integer action) {
        agent_host.sendCommand((String) actionSpace.encode(action));

        last_world_state = waitForObservationsAndRewards();
        last_observation = observationSpace.getObservation(last_world_state);

        if (isDone()) {
            logger.info("Mission ended");
        }

        return new StepReply<MalmoBox>(last_observation, getRewards(last_world_state), isDone(), null);
    }

    private double getRewards(WorldState world_state) {
        double rval = 0;

        for (int i = 0; i < world_state.getRewards().size(); i++) {
            TimestampedReward reward = world_state.getRewards().get(i);
            rval += reward.getValue();
        }

        return rval;
    }

    @Override
    public boolean isDone() {
        return !last_world_state.getIsMissionRunning();
    }

    @Override
    public MDP<MalmoBox, Integer, DiscreteSpace> newInstance() {
        return new MalmoEnv(mission, missionRecord, actionSpace, observationSpace, framePolicy, clientPool);
    }
}
