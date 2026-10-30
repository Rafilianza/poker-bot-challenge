"""Strict PHH replay, native observations, hidden information, and profit labels."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from arena.bc_features import FEATURE_COUNT, action_options, action_target, encode_observation
from arena.phh import _pokerkit, convert_hand


def river_hand():
    return {
        'variant': 'NT', 'antes': [0] * 4, 'blinds_or_straddles': [50, 100, 0, 0],
        'min_bet': 100, 'starting_stacks': [10000] * 4, 'hand': 7,
        'players': ['Pluribus', 'Opponent', 'Third', 'Fourth'],
        'actions': [
            'd dh p1 AsAh', 'd dh p2 KdKh', 'd dh p3 QcQh', 'd dh p4 7c8c',
            'p3 f', 'p4 f', 'p1 cbr 250', 'p2 cc',
            'd db 2c3d4h', 'p1 cc', 'p2 cbr 500', 'p1 cc',
            'd db 9s', 'p1 cbr 1000', 'p2 cc',
            'd db 2h', 'p1 cc', 'p2 cc', 'p1 sm AsAh', 'p2 sm',
        ],
        'finishing_stacks': [11750, 8250, 10000, 10000],
    }


def convert(hand, **kwargs):
    return convert_hand(hand, source_id='hand-content-digest', group_id='source-session-3',
                        source_namespace='fixture-collection', **kwargs)


class PHHConverterTests(unittest.TestCase):
    def assert_converted(self, result):
        self.assertIsNone(result['skip_reason'], result['skip_detail'])
        self.assertTrue(result['records'])
        self.assertTrue(all(type(value) is int for value in result['counters'].values()))

    def test_observations_match_the_actual_sdk_engine_on_every_street(self):
        from macpoker.actions import Action
        from macpoker.cards import parse_card
        from macpoker.hand import Sink, play_hand

        converted = convert(river_hand(), teacher_name='*')
        self.assert_converted(converted)
        records = converted['records']

        class ReplaySink(Sink):
            def __init__(self):
                self.observations = []

            def request_action(self, seat, view):
                record = records[len(self.observations)]
                self.observations.append(deepcopy(view))
                action = record['executed_action']
                return Action(action['action'], action['amount'])

        sink = ReplaySink()
        card_names = ['As', 'Ah', 'Kd', 'Kh', 'Qc', 'Qh', '7c', '8c',
                      '2c', '3d', '4h', '9s', '2h']
        deck = [parse_card(card) for card in card_names]
        native_result = play_hand([200] * 4, 3, deck, sink, 1, 2, hand_no=7)
        self.assertEqual(native_result.deltas, [35, -35, 0, 0])
        self.assertEqual(len(records), len(sink.observations))
        for record, expected in zip(records, sink.observations):
            expected.update(players=[0, 1, 2, 3], clock_ms=0)
            self.assertEqual(record['observation'], expected)
            self.assertEqual(len(encode_observation(expected)), FEATURE_COUNT)
            target, error = action_target(expected, record['executed_action'])
            self.assertTrue(action_options(expected)[1][target])
            self.assertGreaterEqual(error, 0)
        self.assertEqual({record['observation']['street'] for record in records},
                         {'preflop', 'flop', 'turn', 'river'})

    def test_known_opponents_and_future_board_never_leak(self):
        original = river_hand()
        changed = deepcopy(original)
        changed['actions'][1] = 'd dh p2 JdJh'
        changed['actions'][2] = 'd dh p3 TcTh'
        changed['actions'][12] = 'd db 8s'
        original_records = convert(original)['records']
        changed_result = convert(changed)
        self.assert_converted(changed_result)
        # Public observations before the altered turn remain identical.
        for first, second in zip(original_records, changed_result['records']):
            if first['observation']['street'] in ('preflop', 'flop'):
                self.assertEqual(first, second)
            observation = first['observation']
            self.assertEqual(observation['hole'], ['As', 'Ah'])
            self.assertEqual(len(observation['board']),
                             {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[observation['street']])
            self.assertNotIn('finishing_stacks', observation)
            self.assertNotIn('reward', observation)
            self.assertNotIn('winnings', first)
            self.assertNotIn('outcome', first)

    def test_private_cards_can_be_recovered_from_that_players_later_show(self):
        hand = river_hand()
        hand['actions'][0] = 'd dh p1 ????'
        recovered = convert(hand)
        self.assert_converted(recovered)
        self.assertEqual(recovered['records'], convert(river_hand())['records'])

    def test_missing_private_cards_skip_the_teacher_instead_of_guessing(self):
        hand = river_hand()
        hand['actions'][2] = 'd dh p3 ????'
        missing = convert(hand, teacher_name='Third')
        self.assertEqual(missing['skip_reason'], 'missing_teacher_cards')
        self.assertEqual(missing['counters']['missing_hole_decisions'], 1)
        self.assertEqual(missing['records'], [])
        # A missing deal can be filled with unknown placeholders, never random cards.
        hand['actions'].pop(2)
        missing = convert(hand, teacher_name='Third')
        self.assertEqual(missing['skip_reason'], 'missing_teacher_cards')
        self.assert_converted(convert(hand))

    def test_fold_check_call_and_raise_labels_use_existing_seven_actions(self):
        result = convert(river_hand(), teacher_name='*')
        self.assert_converted(result)
        records = result['records']
        targets = [action_target(record['observation'], record['executed_action'])[0]
                   for record in records]
        self.assertEqual(targets[0:2], [0, 0])
        self.assertEqual(records[0]['executed_action'], {'action': 'fold', 'amount': 0})
        self.assertTrue(all(0 <= target < 7 for target in targets))
        call = next(record for record in records if record['executed_action']['action'] == 'call')
        self.assertEqual(call['executed_action']['amount'], call['observation']['to_call'])
        self.assertEqual(call['observation']['history'][-1], ['preflop', 0, 'raise', 5.0])

    def test_whole_winning_player_hand_filter_preserves_identical_records(self):
        hand = river_hand()
        all_records = convert(hand, teacher_name='*')
        profitable = convert(hand, teacher_name='*', profitable_only=True)
        self.assert_converted(profitable)
        expected = [record for record in all_records['records'] if record['teacher']['player'] == 0]
        self.assertEqual(profitable['records'], expected)
        self.assertEqual(len(expected), 5)
        self.assertEqual(profitable['counters']['excluded_unprofitable_decisions'],
                         len(all_records['records']) - len(expected))
        self.assertEqual(convert(hand)['records'], convert(hand, profitable_only=True)['records'])
        loser = convert(hand, teacher_name='Opponent', profitable_only=True)
        self.assertEqual(loser['skip_reason'], 'no_profitable_teacher_decisions')
        tied = convert(hand, teacher_name='Third', profitable_only=True)
        self.assertEqual(tied['records'], [])

    def test_multiple_selected_winners_are_all_retained(self):
        hand = river_hand()
        hand['players'][1] = 'Pluribus'
        hand['actions'] = hand['actions'][:4] + [
            'p3 cbr 200', 'p4 f', 'p1 cc', 'p2 cc',
            'd db 2c3d4h', 'p1 cc', 'p2 cc', 'p3 cc',
            'd db 5s', 'p1 cbr 400', 'p2 cc', 'p3 f',
            'd db 6c', 'p1 cc', 'p2 cc', 'p1 sm AsAh', 'p2 sm KdKh',
        ]
        hand['finishing_stacks'] = [10100, 10100, 9800, 10000]
        result = convert(hand, profitable_only=True)
        self.assert_converted(result)
        self.assertEqual({record['teacher']['player'] for record in result['records']}, {0, 1})
        self.assertEqual(result['records'], convert(hand)['records'])

    def test_results_are_required_consistent_and_complete_for_profit_mode(self):
        missing = river_hand()
        missing.pop('finishing_stacks')
        self.assertEqual(convert(missing, profitable_only=True)['skip_reason'], 'missing_outcome')
        self.assert_converted(convert(missing))
        inconsistent = river_hand()
        inconsistent['finishing_stacks'][0] += 1
        self.assertEqual(convert(inconsistent)['skip_reason'], 'inconsistent_outcome')
        incomplete = river_hand()
        incomplete['actions'] = incomplete['actions'][:9]
        result = convert(incomplete, profitable_only=True)
        self.assertEqual(result['skip_reason'], 'incomplete_hand')
        self.assertEqual(result['records'], [])
        unknown_live = river_hand()
        unknown_live['actions'][1] = 'd dh p2 ????'
        unknown_live['actions'].pop()
        self.assertEqual(convert(unknown_live)['skip_reason'], 'unresolved_showdown')

    def test_real_pluribus_fractional_tie_keeps_exact_source_results(self):
        # A reduced copy of public Pluribus session 102, hand 0. The tied
        # 1,349-chip pot pays 674.5 each, rather than assigning an odd chip.
        hand = {
            'variant': 'NT', 'ante_trimming_status': True,
            'antes': [0] * 6, 'blinds_or_straddles': [50, 100, 0, 0, 0, 0],
            'min_bet': 100, 'starting_stacks': [10000] * 6,
            'players': ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus'],
            'actions': [
                'd dh p1 2cAc', 'd dh p2 9sTc', 'd dh p3 4c6s',
                'd dh p4 Ks2d', 'd dh p5 6dAd', 'd dh p6 Jh5d',
                'p3 f', 'p4 f', 'p5 cbr 225', 'p6 f', 'p1 cc', 'p2 cc',
                'd db Qs9c4s', 'p1 cc', 'p2 cc', 'p5 cc',
                'd db As', 'p1 cc', 'p2 cc', 'p5 cc',
                'd db 8d', 'p1 cc', 'p2 cc', 'p5 cbr 337', 'p1 cc', 'p2 f',
                'p5 sm 6dAd', 'p1 sm 2cAc',
            ],
            'finishing_stacks': [10112.5, 9775.0, 10000.0, 10000.0, 10112.5, 10000.0],
        }
        result = convert_hand(hand, source_id='102-0', group_id='102',
                              source_namespace='pluribus', teacher_name='*', profitable_only=True)
        self.assert_converted(result)
        self.assertEqual({record['teacher']['player'] for record in result['records']}, {0, 4})
        hand['finishing_stacks'][0] += 0.5
        invalid = convert_hand(hand, source_id='102-0', group_id='102', source_namespace='pluribus')
        self.assertEqual(invalid['skip_reason'], 'inconsistent_outcome')

    def test_mismatching_loaded_pokerkit_version_is_rejected(self):
        _pokerkit()
        with patch('arena.phh.metadata.distributions', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'requires PokerKit 0.7.6'):
                _pokerkit()

    def test_contradictory_partial_show_is_not_used_to_rewrite_private_cards(self):
        hand = river_hand()
        hand['actions'][-2] = 'p1 sm Ad'
        result = convert(hand)
        self.assertEqual(result['skip_reason'], 'invalid_replay')
        self.assertIn('disagree', result['skip_detail'])

    def test_winnings_are_collected_pots_and_never_mistaken_for_net_profit(self):
        hand = river_hand()
        hand['winnings'] = [3500, 0, 0, 0]
        self.assert_converted(convert(hand, profitable_only=True))
        hand['winnings'][0] = 1750
        self.assertEqual(convert(hand)['skip_reason'], 'inconsistent_outcome')

    def test_out_of_turn_and_missing_betting_actions_are_not_repaired(self):
        hand = river_hand()
        hand['actions'].pop(4)
        result = convert(hand)
        self.assertEqual(result['skip_reason'], 'invalid_replay')
        self.assertIn('out of turn', result['skip_detail'])
        self.assertEqual(result['records'], [])

    def test_short_all_in_raise_does_not_reopen_and_all_in_has_its_own_label(self):
        hand = river_hand()
        hand['actions'] = hand['actions'][:4] + [
            'p3 f', 'p4 f', 'p1 cbr 9000', 'p2 cbr 10000', 'p1 cc',
            'p1 sm AsAh', 'p2 sm KdKh', 'd db 2c3d4h', 'd db 9s', 'd db 2h',
        ]
        hand['finishing_stacks'] = [20000, 0, 10000, 10000]
        result = convert(hand, teacher_name='*')
        self.assert_converted(result)
        all_in = result['records'][-2]
        self.assertEqual(all_in['observation']['min_raise_to'], 200)
        self.assertEqual(action_target(all_in['observation'], all_in['executed_action']), (6, 0))
        response = result['records'][-1]
        self.assertEqual(response['executed_action'], {'action': 'call', 'amount': 20})
        self.assertFalse(response['observation']['can_raise'])
        self.assertEqual(action_options(response['observation'])[1],
                         [True, True, False, False, False, False, False])

    def test_fractional_normalization_is_retained_and_error_is_in_big_blinds(self):
        hand = river_hand()
        hand['actions'][6] = 'p1 cbr 225'
        hand['finishing_stacks'] = [11725, 8275, 10000, 10000]
        result = convert(hand)
        self.assert_converted(result)
        first = result['records'][0]
        self.assertEqual(first['executed_action']['amount'], 4.5)
        self.assertEqual(result['raise_errors_bb'][0],
                         action_target(first['observation'], first['executed_action'])[1] / 2)
        call = convert(hand, teacher_name='Opponent')['records'][0]
        self.assertEqual(call['executed_action']['amount'], 2.5)

    def test_teacher_identity_survives_different_hands_and_rotated_seats(self):
        first = convert(river_hand())
        second = convert_hand(river_hand(), source_id='another-content-digest',
                              group_id='another-session', source_namespace='fixture-collection')
        self.assertEqual(first['teachers'], second['teachers'])
        self.assertNotEqual(first['records'][0]['decision_id'], second['records'][0]['decision_id'])
        rotated = river_hand()
        rotated['players'] = ['Opponent', 'Pluribus', 'Third', 'Fourth']
        self.assertEqual(first['teachers'], convert(rotated)['teachers'])
        self.assertEqual(first, convert(river_hand()))

    def test_explicit_unsupported_and_invalid_input_reasons(self):
        for change, expected in [
            ({'variant': 'PO'}, 'unsupported_variant'),
            ({'starting_stacks': [10000] * 3}, 'unsupported_table_size'),
            ({'starting_stacks': [float('inf')] * 4}, 'unsupported_stacks'),
            ({'starting_stacks': [10000, 9000, 10000, 10000]}, 'unsupported_stack_depth'),
            ({'antes': [1] * 4}, 'unsupported_antes'),
            ({'blinds_or_straddles': [50, 100, 200, 0]}, 'unsupported_blinds'),
            ({'min_bet': 50}, 'unsupported_min_bet'),
            ({'players': None}, 'missing_players'),
            ({'finishing_stacks': [11750, float('nan'), 10000, 10000]}, 'invalid_outcome'),
        ]:
            with self.subTest(change=change):
                self.assertEqual(convert({**river_hand(), **change})['skip_reason'], expected)
        self.assertEqual(convert(river_hand(), teacher_name='Absent')['skip_reason'], 'teacher_not_present')


if __name__ == '__main__':
    unittest.main()
