//! State machine for robot behavior management
//!
//! A simple but flexible state machine implementation for managing
//! robot states and transitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

/// A state in the state machine
pub trait State<C, E>: Send + Sync {
    /// Called when entering this state
    fn on_enter(&mut self, _context: &mut C) {}

    /// Called every update tick while in this state
    fn on_update(&mut self, _context: &mut C, _dt: f64) {}

    /// Called when exiting this state
    fn on_exit(&mut self, _context: &mut C) {}

    /// Handle an event and optionally return a new state ID
    fn handle_event(&mut self, _context: &mut C, _event: &E) -> Option<String> {
        None
    }

    /// Get the state name/ID
    fn name(&self) -> &str;
}

/// A transition between states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Source state
    pub from: String,
    /// Target state
    pub to: String,
    /// Optional event that triggers this transition
    pub event: Option<String>,
}

impl Transition {
    /// Create a new transition
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            event: None,
        }
    }

    /// Create a transition triggered by an event
    #[must_use]
    pub fn on_event(
        from: impl Into<String>,
        event: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            event: Some(event.into()),
        }
    }
}

/// A simple state machine
pub struct StateMachine<C, E> {
    states: Vec<Box<dyn State<C, E>>>,
    state_indices: HashMap<String, usize>,
    current_state: String,
    current_handler_idx: Option<usize>,
    transitions: Vec<Transition>,
}

impl<C, E> StateMachine<C, E> {
    /// Create a new state machine with an initial state
    pub fn new(initial_state: impl Into<String>) -> Self {
        let initial = initial_state.into();
        Self {
            states: Vec::new(),
            state_indices: HashMap::new(),
            current_state: initial,
            current_handler_idx: None,
            transitions: Vec::new(),
        }
    }

    /// Add a state to the machine
    pub fn add_state<S: State<C, E> + 'static>(&mut self, state: S) {
        let name = state.name().to_string();
        let idx = self.states.len();
        self.states.push(Box::new(state));
        self.state_indices.insert(name.clone(), idx);
        // Update cached index if this is the current state
        if name == self.current_state {
            self.current_handler_idx = Some(idx);
        }
    }

    /// Add a transition
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Get the current state name
    #[inline]
    #[must_use]
    pub fn current_state(&self) -> &str {
        &self.current_state
    }

    /// Check if in a specific state
    #[inline]
    #[must_use]
    pub fn is_in_state(&self, state: &str) -> bool {
        self.current_state == state
    }

    /// Transition to a new state
    pub fn transition_to(&mut self, new_state: impl Into<String>, context: &mut C) -> bool {
        let new_state = new_state.into();

        let new_idx = match self.state_indices.get(&new_state) {
            Some(&idx) => idx,
            None => {
                tracing::warn!("Attempted to transition to unknown state: {}", new_state);
                return false;
            }
        };

        // Exit current state
        if let Some(idx) = self.current_handler_idx {
            self.states[idx].on_exit(context);
        }

        // Enter new state
        self.current_state = new_state.clone();
        self.current_handler_idx = Some(new_idx);
        self.states[new_idx].on_enter(context);

        tracing::debug!("State transition: -> {}", new_state);
        true
    }

    /// Update the current state
    pub fn update(&mut self, context: &mut C, dt: f64) {
        if let Some(idx) = self.current_handler_idx {
            self.states[idx].on_update(context, dt);
        }
    }

    /// Handle an event
    pub fn handle_event(&mut self, context: &mut C, event: &E) -> bool {
        // First let the current state handle it
        if let Some(idx) = self.current_handler_idx {
            if let Some(new_state) = self.states[idx].handle_event(context, event) {
                return self.transition_to(new_state, context);
            }
        }
        false
    }

    /// Initialize the state machine (call on_enter for initial state)
    pub fn initialize(&mut self, context: &mut C) {
        if self.current_handler_idx.is_none() {
            self.current_handler_idx = self.state_indices.get(&self.current_state).copied();
        }
        if let Some(idx) = self.current_handler_idx {
            self.states[idx].on_enter(context);
        }
    }
}

/// A simple function-based state for quick prototyping
pub struct FnState<C, E> {
    name: String,
    on_enter: Option<Box<dyn Fn(&mut C) + Send + Sync>>,
    on_update: Option<Box<dyn Fn(&mut C, f64) + Send + Sync>>,
    on_exit: Option<Box<dyn Fn(&mut C) + Send + Sync>>,
    on_event: Option<Box<dyn Fn(&mut C, &E) -> Option<String> + Send + Sync>>,
}

impl<C, E> FnState<C, E> {
    /// Create a new function-based state
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            on_enter: None,
            on_update: None,
            on_exit: None,
            on_event: None,
        }
    }

    /// Set the on_enter callback
    #[must_use]
    pub fn with_enter<F: Fn(&mut C) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_enter = Some(Box::new(f));
        self
    }

    /// Set the on_update callback
    #[must_use]
    pub fn with_update<F: Fn(&mut C, f64) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_update = Some(Box::new(f));
        self
    }

    /// Set the on_exit callback
    #[must_use]
    pub fn with_exit<F: Fn(&mut C) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_exit = Some(Box::new(f));
        self
    }

    /// Set the on_event callback
    #[must_use]
    pub fn with_event<F: Fn(&mut C, &E) -> Option<String> + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_event = Some(Box::new(f));
        self
    }
}

impl<C, E> State<C, E> for FnState<C, E> {
    fn on_enter(&mut self, context: &mut C) {
        if let Some(ref f) = self.on_enter {
            f(context);
        }
    }

    fn on_update(&mut self, context: &mut C, dt: f64) {
        if let Some(ref f) = self.on_update {
            f(context, dt);
        }
    }

    fn on_exit(&mut self, context: &mut C) {
        if let Some(ref f) = self.on_exit {
            f(context);
        }
    }

    fn handle_event(&mut self, context: &mut C, event: &E) -> Option<String> {
        if let Some(ref f) = self.on_event {
            f(context, event)
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// An enum-based state machine for compile-time safety
pub struct EnumStateMachine<S, C>
where
    S: Clone + Eq + Hash,
{
    handlers: Vec<Box<dyn Fn(&mut C, f64) + Send + Sync>>,
    state_indices: HashMap<S, usize>,
    current: S,
    current_handler_idx: Option<usize>,
}

impl<S: Clone + Eq + Hash, C> EnumStateMachine<S, C> {
    /// Create a new enum state machine
    #[must_use]
    pub fn new(initial: S) -> Self {
        Self {
            handlers: Vec::new(),
            state_indices: HashMap::new(),
            current: initial,
            current_handler_idx: None,
        }
    }

    /// Add a state handler
    #[must_use]
    pub fn add_state<F: Fn(&mut C, f64) + Send + Sync + 'static>(
        mut self,
        state: S,
        handler: F,
    ) -> Self {
        let idx = self.handlers.len();
        self.handlers.push(Box::new(handler));
        let is_current = state == self.current;
        self.state_indices.insert(state, idx);
        if is_current {
            self.current_handler_idx = Some(idx);
        }
        self
    }

    /// Get the current state
    #[inline]
    #[must_use]
    pub fn current(&self) -> &S {
        &self.current
    }

    /// Transition to a new state
    pub fn set_state(&mut self, state: S) {
        self.current_handler_idx = self.state_indices.get(&state).copied();
        self.current = state;
    }

    /// Update the current state
    pub fn update(&mut self, context: &mut C, dt: f64) {
        if let Some(idx) = self.current_handler_idx {
            self.handlers[idx](context, dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestContext {
        counter: i32,
        entered: Vec<String>,
        exited: Vec<String>,
    }

    #[derive(Debug)]
    enum TestEvent {
        Start,
        Stop,
    }

    #[test]
    fn test_state_machine_basic() {
        let mut sm: StateMachine<TestContext, TestEvent> = StateMachine::new("idle");

        sm.add_state(
            FnState::<TestContext, TestEvent>::new("idle")
                .with_enter(|ctx: &mut TestContext| ctx.entered.push("idle".into()))
                .with_exit(|ctx: &mut TestContext| ctx.exited.push("idle".into())),
        );

        sm.add_state(
            FnState::<TestContext, TestEvent>::new("running")
                .with_enter(|ctx: &mut TestContext| ctx.entered.push("running".into()))
                .with_update(|ctx: &mut TestContext, _dt| ctx.counter += 1),
        );

        let mut ctx = TestContext::default();
        sm.initialize(&mut ctx);

        assert_eq!(sm.current_state(), "idle");
        assert_eq!(ctx.entered, vec!["idle"]);

        sm.transition_to("running", &mut ctx);
        assert_eq!(sm.current_state(), "running");
        assert_eq!(ctx.exited, vec!["idle"]);
        assert_eq!(ctx.entered, vec!["idle", "running"]);

        sm.update(&mut ctx, 0.01);
        assert_eq!(ctx.counter, 1);
    }

    #[test]
    fn test_state_machine_event() {
        let mut sm: StateMachine<TestContext, TestEvent> = StateMachine::new("idle");

        sm.add_state(FnState::<TestContext, TestEvent>::new("idle").with_event(
            |_ctx: &mut TestContext, event: &TestEvent| match event {
                TestEvent::Start => Some("running".into()),
                _ => None,
            },
        ));

        sm.add_state(FnState::<TestContext, TestEvent>::new("running"));

        let mut ctx = TestContext::default();
        sm.initialize(&mut ctx);

        assert_eq!(sm.current_state(), "idle");

        sm.handle_event(&mut ctx, &TestEvent::Start);
        assert_eq!(sm.current_state(), "running");
    }

    #[test]
    fn test_enum_state_machine() {
        #[derive(Clone, Eq, PartialEq, Hash)]
        enum RobotState {
            Idle,
            Walking,
        }

        struct Robot {
            speed: f64,
        }

        let mut sm = EnumStateMachine::<RobotState, Robot>::new(RobotState::Idle)
            .add_state(RobotState::Idle, |robot: &mut Robot, _dt| {
                robot.speed = 0.0;
            })
            .add_state(RobotState::Walking, |robot: &mut Robot, dt| {
                robot.speed = 1.0 * dt;
            });

        let mut robot = Robot { speed: 0.0 };

        sm.update(&mut robot, 0.1);
        assert_eq!(robot.speed, 0.0);

        sm.set_state(RobotState::Walking);
        sm.update(&mut robot, 0.1);
        assert!((robot.speed - 0.1).abs() < 0.001);
    }
}
